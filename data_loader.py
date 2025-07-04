# Import required libraries
import pyodbc                          # For connecting to SQL Server
import pandas as pd                   # For working with tabular data
import time                           # For measuring execution time
import os                             # For file operations
from decimal import Decimal, ROUND_HALF_UP  # For precise rounding of floats

def round_all_float_columns(df, decimals=2):
    """
    Round all float columns in the DataFrame to a fixed number of decimal places
    using ROUND_HALF_UP (standard rounding method).
    """
    float_cols = df.select_dtypes(include=['float']).columns  # Identify float columns
    for col in float_cols:
        df[col] = df[col].apply(
            lambda x: float(Decimal(str(x)).quantize(Decimal(f'1.{"0"*decimals}'), rounding=ROUND_HALF_UP))
            if pd.notnull(x) else x  # Skip NaNs
        )
    return df

def transform_dataframe(df, keep_float64=None):
    """
    Optimize memory usage:
    - Downcast numeric columns to float32/int32 (except those explicitly kept as float64).
    - Convert object columns with low cardinality to 'category' type.
    """
    if keep_float64 is None:
        keep_float64 = []

    # Downcast numeric columns not in keep_float64
    for col in df.select_dtypes(include='number').columns:
        if col not in keep_float64:
            df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert low-cardinality object columns to 'category' dtype
    for col in df.select_dtypes(include='object').columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    return df

def load_input_data(server, database, table, status_zapisa,
                    id_var, schema,
                    trusted_connection='yes', chunksize=50000,
                    parquet_path=None):
    """
    Load data from a SQL Server table (including _edi), selecting only rows with status_zapisa < threshold.
    - Keeps only the latest record per id_var using ROW_NUMBER().
    - Rounds all float values to 2 decimals.
    - Optimizes memory.
    - Optionally saves the result to a Parquet file.

    Parameters:
    - server, database: SQL Server connection info
    - table: Base table name (EDI variant assumed as table_edi - table for changed records)
    - status_zapisa: record version ( 1 in base table; 2, 3, 4, 5... - in _edi table)
    - id_var: ID variable
    - schema: Schema name
    - trusted_connection: Use Windows Auth (default yes)
    - chunksize: Rows to load per chunk (default 50,000)
    - parquet_path: Optional path to save the final DataFrame as a Parquet file
    """
    
    # SQL query:
    # 1. Combines base and _edi tables
    # 2. Filters by status_zapisa
    # 3. Applies ROW_NUMBER() to keep latest record per id_var
    query = f"""
    SELECT *, FORMAT({id_var}, '000000000') AS ident
    FROM (
        SELECT *, 
               ROW_NUMBER() OVER (PARTITION BY {id_var} ORDER BY status_zapisa DESC) AS rn
        FROM (
            SELECT * 
            FROM {schema}.{table}
            WHERE status_zapisa < {status_zapisa}

            UNION ALL

            SELECT * 
            FROM {schema}.{table}_edi
            WHERE status_zapisa < {status_zapisa}
        ) AS combined
    ) AS ranked
    WHERE rn = 1;
    """

    # Construct connection string
    conn_str = (
        f'DRIVER={{SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'Trusted_Connection={trusted_connection};'
    )

    try:
        start = time.time()  # Start timing the operation

        # Open a connection and read the SQL query in chunks
        with pyodbc.connect(conn_str) as conn:
            chunks = pd.read_sql(query, conn, chunksize=chunksize)

            all_chunks = []  # List to collect processed chunks

            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Round float columns to 2 decimals
                chunk = round_all_float_columns(chunk, decimals=2)

                # Identify float columns to keep at float64
                float_cols = chunk.select_dtypes(include=['float']).columns.tolist()

                # Optimize the rest of the dataframe
                chunk = transform_dataframe(chunk, keep_float64=float_cols)

                print(f"Chunk {i+1}: {chunk.shape[0]} rows, ~{chunk.memory_usage(deep=True).sum() / 1e6:.2f} MB")
                all_chunks.append(chunk)

                if i == 0:
                    print("Preview first chunk:")
                    print(chunk.head())

        # Combine all processed chunks
        input_data = pd.concat(all_chunks, ignore_index=True)
        end = time.time()

        print(f"✅ Loaded {input_data.shape[0]} rows from SQL in {end - start:.2f} seconds.")

        # Optional: save to Parquet format
        if parquet_path:
            input_data.to_parquet(parquet_path, index=False)
            file_size = os.path.getsize(parquet_path) / 1e6
            print(f"📦 Saved to '{parquet_path}' ({file_size:.2f} MB)")

        return input_data

    except Exception as e:
        # Handle errors during the data loading process
        print("❌ Error while executing the query:", e)
        return None

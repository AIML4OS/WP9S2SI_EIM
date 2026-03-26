from sqlalchemy import create_engine, text, inspect, event
import urllib
import time 

def save_output_data(df, server, database, table_output, schema='dbo',
                     trusted_connection='yes', dtype=None,
                     delete_before_insert=False):
    """
    Saves a DataFrame to a SQL Server table using SQLAlchemy.
    Optionally deletes existing rows where STATUS_ZAPISA matches any value
    found in the new DataFrame before inserting.

    Args:
        df (pd.DataFrame): Data to write to SQL Server.
        server (str): SQL Server name.
        database (str): Target database name.
        table_output (str): Target table name where data will be saved.
        schema (str): Target schema (default 'dbo').
        trusted_connection (str): Use trusted connection (default 'yes').
        dtype (dict or None): Optional SQL data types for columns.
        delete_before_insert (bool): If True, delete existing rows with STATUS_ZAPISA (=record version) values found in df.
    """
    try:
        conn_str = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection={trusted_connection};"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}", fast_executemany=True)

        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                if executemany:
                    cursor.fast_executemany = True
 
        time_start = time.time()
        with engine.begin() as connection:
            inspector = inspect(engine)
            existing_columns = inspector.get_columns(table_output, schema=schema)
            existing_col_names = [col['name'] for col in existing_columns]

            # Keep only columns that exist in the target table
            df_filtered = df.loc[:, df.columns.intersection(existing_col_names)]

            if delete_before_insert:
                # Find STATUS_ZAPISA column case-insensitive
                status_col = next((col for col in df_filtered.columns if col.lower() == 'status_zapisa'), None)
                if status_col is None:
                    raise ValueError("STATUS_ZAPISA column not found in DataFrame.")

                unique_statuses = df_filtered[status_col].unique().tolist()
                if unique_statuses:
                    placeholders = ", ".join([f":status_{i}" for i in range(len(unique_statuses))])
                    delete_query = text(f"DELETE FROM {schema}.{table_output} WHERE STATUS_ZAPISA IN ({placeholders})")
                    params = {f"status_{i}": status for i, status in enumerate(unique_statuses)}

                    result = connection.execute(delete_query, params)
                    print(f"🗑 Deleted {result.rowcount} rows from {schema}.{table_output} where STATUS_ZAPISA in {unique_statuses}")

            # Insert data
            df_filtered.to_sql(
                name=table_output,
                con=connection,
                schema=schema,
                if_exists='append',
                index=False,
                dtype=dtype,
                chunksize=50000)
        
        time_end = time.time()
        print(f"✅ Data successfully written to {schema}.{table_output} ({df_filtered.shape[0]} rows) {time_end - time_start:.2f} seconds.")

    except Exception as e:
        print("❌ Failed to save data to SQL Server:", e)


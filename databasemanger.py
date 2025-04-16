import os
from pathlib import Path
import chromadb
import pandas as pd
import sqlite3
import ollama
from typing import Dict, Any, List
from disk_manager import manage_vector_store,save_vector_data_to_disk,load_vector_data_from_disk

class DatabaseManager:
    _instances = {}  

    def __new__(cls, db_name: str):
        if db_name not in cls._instances:
            cls._instances[db_name] = super(DatabaseManager, cls).__new__(cls)
            cls._instances[db_name].init(db_name)  
        return cls._instances[db_name]

    def init(self, db_name: str):
        self.db_name = db_name
        self.set_path()
        self.client = chromadb.PersistentClient(path=str(self.chromadb_path)) 


    def set_path(self):
        base_dir = Path("Data/dev_databases") 
        self.sqlite_path = base_dir / self.db_name / f"{self.db_name}.sqlite"
        self.table_description_path = base_dir / self.db_name / "database_description"
        self.chromadb_path = base_dir / self.db_name / f"{self.db_name}.chromadb"
        print(f"SQLite Path: {self.sqlite_path}")
        print(f"Table Description Path: {self.table_description_path}")

    def convert_chromadb_name(self, database_name: str, table_name: str, column_names: list):
        converted_names = {}
        for i, column in enumerate(column_names, start=1):
            converted_name = f"{database_name}_{table_name}_column{i}"
            converted_names[column] = converted_name
        return converted_names

    def save_vectordb(self):
        try:                
            df_dict = self.get_data_to_embed()
            table_names = self.get_table_description_filenames()
            
            with open('database_progress.txt', 'a', encoding='utf-8') as f:
                for table_name in table_names:
                    table_name = table_name.replace(".csv", "")
                    f.write(f"Embedding table '{table_name}'...\n")
                    f.flush() 
                    
                    print(f"Embedding table '{table_name}'...")
                    df = df_dict[table_name]
                    columns_to_embed = df.columns.tolist()
                    vector_store_filename_dict = self.convert_chromadb_name(self.db_name, table_name, columns_to_embed)

                    for column in columns_to_embed:
                        if column in df.columns:
                            print(f"Embedding column '{column}'...")
                            column_data = df[column].fillna('null').to_string(index=False).split('\n')

                            column_name = vector_store_filename_dict[column]
                            vector_store = manage_vector_store(self.client, column_name)
                            embeddings, metadata, ids, docs = [], [], [], []
                            
                            for idx, out_string in enumerate(column_data):
                                print(f"Embedding idx '{idx}'...")
                                embedding = self._embedding_function(out_string)
                                embeddings.append(embedding)
                                metadata.append({'index': idx, 'text': out_string})
                                ids.append(f"id{idx}")
                                docs.append(str(df.iloc[idx].to_list()))
                            
                            if vector_store:
                                vector_store.add(embeddings=embeddings, metadatas=metadata, ids=ids, documents=docs)

                            print(f"Column '{column}' has been embedded and stored in ChromaDB.")
                        else:
                            print(f"Column '{column}' does not exist in the CSV file.")
        except Exception as e:
            print(f"Error saving vector database: {e}")

    def get_table_description(self) -> Dict[str, Dict[str, Any]]:
        """
        Save the information of each table in the table_description folder as corresponding dictionaries.
        """
        descriptions = {}
        for file_path in self.table_description_path.glob("*.csv"):
            table_name = file_path.stem
            descriptions[table_name] = self._parse_csv(file_path)
        
        print("Table descriptions extracted successfully.")
        return descriptions

    def get_primary_foreign_keys(self) -> Dict[str, Any]:
        """
        Retrieve primary key and foreign key information for each table, including the relationships between primary and foreign keys.
        """
        keys_info = {}

        with sqlite3.connect(self.sqlite_path) as connection:
            cursor = connection.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns_info = cursor.fetchall()

                primary_keys = [col[1] for col in columns_info if col[5] == 1] 
                keys_info[table_name] = {
                    "primary_keys": primary_keys,
                    "foreign_keys": [],
                    "foreign_key_relations": {}
                }

                cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
                foreign_keys_info = cursor.fetchall()
                for fk in foreign_keys_info:
                    foreign_key_column = fk[3] 
                    referenced_table = fk[2]   
                    referenced_column = fk[4] 

                    keys_info[table_name]["foreign_keys"].append(foreign_key_column)
                    
                    if referenced_table not in keys_info[table_name]["foreign_key_relations"]:
                        keys_info[table_name]["foreign_key_relations"][referenced_table] = []
                    keys_info[table_name]["foreign_key_relations"][referenced_table].append(referenced_column)

        print("Primary and foreign keys extracted successfully.")
        
        primary_keys = {
            table: info['primary_keys']
            for table, info in keys_info.items()
        }
        
        foreign_keys = []
        for table, info in keys_info.items():
            for ref_table, ref_cols in info['foreign_key_relations'].items():
                for fk, pk in zip(info['foreign_keys'], ref_cols):
                    foreign_keys.append(f'{table}."{fk}"={ref_table}."{pk}"')
        if self.db_name == "debit_card_specializing":
            primary_keys["customers"] = ["CustomerID"]
            primary_keys["gasstations"] = ["GasStationID"]
            primary_keys["products"] = ["ProductID"]
            primary_keys["transactions_1k"] = ["TransactionID"]
            primary_keys["yearmonth"] = ["Date"]
            foreign_keys.append('customers."CustomerID"=transactions_1k."CustomerID"')
            foreign_keys.append('gasstations."GasStationID"=transactions_1k."GasStationID"')
            foreign_keys.append('products."ProductID"=transactions_1k."ProductID"')
            foreign_keys.append('yearmonth."Date"=transactions_1k."Date"')
            foreign_keys.append('customers."CustomerID"=yearmonth."CustomerID"')

        if self.db_name =='european_football_2':
            primary_keys["Player"] = ["id"]
            primary_keys["Team"] = ["id"]
            primary_keys["Match"] = ["id"]
            primary_keys["League"] = ["id"]
            primary_keys["Player_Attributes"] = ["id"]
            primary_keys["Team_Attributes"] = ["id"]
            primary_keys['Country'] = ['id']
            foreign_keys.append('Player_Attributes."player_api_id" = Player."player_api_id"')
            foreign_keys.append('Team_Attributes."team_api_id" = Team."team_api_id"')
            foreign_keys.append('League."country_id" = Country."id"')
            foreign_keys.append('Match."league_id" = League."id"')
            foreign_keys.append('Match."home_team_api_id" = Team."team_api_id"')
            foreign_keys.append('Match."away_team_api_id" = Team."team_api_id"')
            foreign_keys.append('Match."country_id" = Country."id"')

        if self.db_name == 'student_club':
            primary_keys["member"] = ["member_id"]
            primary_keys["major"] = ["major_id"]
            primary_keys["event"] = ["event_id"]
            primary_keys["zip_code"] = ["zip_code"]
            primary_keys["attendance"] = ["link_to_event"]
            primary_keys["budget"] = ["budget_id"]
            primary_keys["expense"] = ["expense_id"]
            primary_keys["income"] = ["income_id"]
            foreign_keys.append('attendance."link_to_member"=member."member_id"')
            foreign_keys.append('attendance."link_to_event"=event."event_id"')
            foreign_keys.append('budget."link_to_event"=event."event_id"')
            foreign_keys.append('expense."link_to_member"=member."member_id"')
            foreign_keys.append('expense."link_to_budget"=budget."budget_id"')
            foreign_keys.append('income."link_to_member"=member."member_id"')
            foreign_keys.append('member."zip"=zip_code."zip_code"')
            foreign_keys.append('member."link_to_major"=major."major_id"')



        return {
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    
    def _embedding_function(self, text: str) -> list:
        """
        Custom embedding function, converting text into vector representation.
        """
        response = ollama.embeddings(
            prompt=text,
            model="mxbai-embed-large"
        )
        return response['embedding']  

    def _parse_csv(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse the CSV file and convert column information into a dictionary.
        """
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']  
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df.to_dict(orient='records')
            except UnicodeDecodeError:
                continue  
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        print(f"Failed to read {file_path} with all attempted encodings")
        return {}
    
    def get_data_to_embed(self) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all tables and their contents from the SQLite database and convert them into DataFrame format.
        """
        dataframes = {}
        
        with sqlite3.connect(self.sqlite_path) as connection:
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]
                    try:
                        table_data = pd.read_sql_query(
                            f'SELECT * FROM "{table_name}";', 
                            connection
                        )
                        for column in table_data.select_dtypes(include=['object']).columns:
                            table_data[column] = table_data[column].apply(
                                lambda x: x.encode('utf-8', 'replace').decode('utf-8') if isinstance(x, str) else x
                            )
                        
                        dataframes[table_name] = table_data
                        
                    except Exception as e:
                        print(f"Error reading table {table_name}: {e}")
                        continue

                return dataframes
                
            except Exception as e:
                print(f"Error fetching data from SQLite: {e}")
                return {}
    
    def get_table_description_filenames(self):
        try:
            files = os.listdir(self.table_description_path)
            file_list = [file for file in files if os.path.isfile(os.path.join(self.table_description_path, file)) 
                        and file != '.DS_Store']
            file_list_final = []
            for file in file_list:
                file = file.replace(".csv", "")
                file_list_final.append(file)
            return file_list_final
        except Exception as e:
            print(f"Error reading table descriptions: {e}")
            return []
        

    def get_table_columns_dict(self):
        data_dict = self.get_data_to_embed()
        actual_tables = set(data_dict.keys())
        
        desc_tables = set(self.get_table_description_filenames())
        
        valid_tables = actual_tables.intersection(desc_tables)
        
        columns_dict = {}
        for table_name in valid_tables:
            columns_dict[f'{table_name}'] = data_dict[table_name].columns.tolist()
                    
        return columns_dict
    
    def create_info_database(self):
        info = {}
        info = self.get_table_columns_dict()
        print(f"{self.sqlite_path}")
        sqlite_path = str(self.sqlite_path).replace(".sqlite", "_info.sqlite")
        
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
            
        with sqlite3.connect(sqlite_path) as connection:
            cursor = connection.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS info (
                            idx INTEGER PRIMARY KEY,
                            table_name TEXT,
                            columns TEXT
                            )''')
            cursor.execute('DELETE FROM info')
            
            idx = 0
            for table_name, columns in info.items():
                print(table_name)
                for i in columns:
                    idx += 1
                    cursor.execute("INSERT INTO info (idx, table_name, columns) VALUES (?,?,?)", 
                                (idx, table_name, str(i)))
            connection.commit()
            print("Database information created successfully.")

    def save_info_vectordb(self):
        try:
            sqlite_path = str(self.sqlite_path).replace(".sqlite", "_info.sqlite")
            with sqlite3.connect(sqlite_path) as connection:
                cursor = connection.cursor()
                
                collections = self.client.list_collections()
                for collection in collections:
                    if collection.name in ['table_info', 'column_info']:
                        self.client.delete_collection(collection.name)
                
                cursor.execute("SELECT DISTINCT table_name FROM info;")
                tables = cursor.fetchall()
                tables = [i[0] for i in tables]
                table_name = "table_info"
                vector_store_table = manage_vector_store(self.client, table_name)

                for idx, table in enumerate(tables):
                    print(f"Embedding table '{str(table)}'...")
                    embeddings, metadata, ids, docs = [], [], [], []
                    embedding = self._embedding_function(str(table))
                    embeddings.append(embedding)
                    metadata.append({'index': idx, 'text': str(table)})
                    ids.append(f"id{idx}")
                    docs.append(str(table))
                    if vector_store_table:    
                        vector_store_table.add(embeddings=embeddings, metadatas=metadata, ids=ids, documents=docs)
                    print(f"Table '{table}' has been embedded and stored in ChromaDB.")

                column_name = "column_info"
                vector_store_column = manage_vector_store(self.client, column_name)
                cursor.execute("SELECT DISTINCT columns FROM info;")
                columns = cursor.fetchall()
                columns = [i[0] for i in columns]
                for idx, column in enumerate(columns):
                    print(f"Embedding column '{str(column)}'...")
                    embeddings, metadata, ids, docs = [], [], [], []
                    embedding = self._embedding_function(str(column))
                    embeddings.append(embedding)
                    metadata.append({'index': idx, 'text': str(column)})
                    ids.append(f"id{idx}")
                    docs.append(str(column))
                    if vector_store_column:    
                        vector_store_column.add(embeddings=embeddings, metadatas=metadata, ids=ids, documents=docs)
                    print(f"Column '{column}' has been embedded and stored in ChromaDB.")
        except Exception as e:
            print(f"Error saving vector database: {e}")


    def create_description_vectordb(self):
        table_descriptions = self.get_table_description()
        table_names = self.get_table_description_filenames()
        
        sqlite_path = str(self.sqlite_path).replace(".sqlite", "_description.sqlite")
        
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
        
        with sqlite3.connect(sqlite_path) as connection:
            cursor = connection.cursor()
        
            for key in table_names:
                safe_table_name = f'{key}'
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_table_name} (
                        "original_column_name" TEXT,
                        "column_name" TEXT,
                        "column_description" TEXT,
                        "data_format" TEXT,
                        "value_description" TEXT
                    )
                ''')
                
                cursor.execute(f'DELETE FROM {safe_table_name}')
                
                for desc in table_descriptions.get(key, []):
                    row = (
                        desc['original_column_name'],
                        desc['column_name'] if pd.notna(desc['column_name']) else None,
                        desc['column_description'] if pd.notna(desc['value_description']) else None,
                        desc['value_description'] if pd.notna(desc['value_description']) else None,
                    )
                    cursor.execute(f'''
                        INSERT INTO {safe_table_name} ("original_column_name", "column_name", "column_description", "value_description")
                        VALUES (?, ?, ?, ?)
                    ''', row)

            connection.commit()

    def save_description_vectordb(self):
        try:
            sqlite_path = str(self.sqlite_path).replace(".sqlite", "_description.sqlite")
            with sqlite3.connect(sqlite_path) as connection:
                cursor = connection.cursor()
                
                tables = self.get_table_description_filenames()
                
                collections = self.client.list_collections()
                for collection in collections:
                    if collection.name.endswith('_description'):
                        self.client.delete_collection(collection.name)
                
                for table in tables:
                    for column in ["original_column_name","column_name","column_description","value_description"]:
                        db_name = f"{table}_{column}_description"
                        vector_db = manage_vector_store(self.client,db_name)
                        column_data = pd.read_sql_query(f'SELECT {column} FROM "{table}";', connection)
                        embeddings, metadata, ids, docs = [], [], [], []
                        for idx,row in column_data.iterrows():
                            out_string = str(row[column])
                            embedding = self._embedding_function(out_string)
                            embeddings.append(embedding)
                            metadata.append({'index': idx, 'text': out_string})
                            ids.append(f"id{idx}")
                            docs.append(str(row.to_list()))
                        if vector_db:
                            vector_db.add(embeddings=embeddings, metadatas=metadata, ids=ids, documents=docs)
                        print(f"Column '{column}' of table '{table}' has been embedded and stored in ChromaDB.")
        except Exception as e:
            print(f"Error saving vector database: {e}")


# if __name__ == '__main__':
#     db_name_list = ["battle_death", "car_1", "concert_singer", "course_teach", "cre_Doc_Template_Mgt", "dog_kennels", "employee_hire_evaluation", "flight_2", "museum_visit", "network_1", "orchestra", "pets_1", "poker_player", "real_estate_properties", "singer", "student_transcripts_tracking", "tvshow", "voter_1"]
#     ["battle_death", "car_1", "concert_singer", "course_teach", "cre_Doc_Template_Mgt", "dog_kennels", "employee_hire_evaluation", "flight_2", "museum_visit", "network_1", "orchestra", "pets_1", "poker_player", "real_estate_properties", "singer", "student_transcripts_tracking", "tvshow", "voter_1", "world_1", "wta_1"]
#     with open('database_progress.txt', 'a', encoding='utf-8') as f:
#         from datetime import datetime
#         f.write(f"\n=== Processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
#         for db_name in db_name_list:
#             db_manager = DatabaseManager(db_name)
#             key_dict = db_manager.get_table_columns_dict()
#             print(key_dict.keys())
            
#             db_manager.save_vectordb()
#             f.write(f"Database '{db_name}' saved successfully.\n")
            
#             db_manager.create_info_database()
#             db_manager.save_info_vectordb()
#             f.write(f"Database information of '{db_name}' saved successfully.\n")
            
#             db_manager.create_description_vectordb()
#             db_manager.save_description_vectordb()
#             f.write(f"Database description of '{db_name}' saved successfully.\n")
            
#             f.write("-" * 50 + "\n")
#             f.flush()
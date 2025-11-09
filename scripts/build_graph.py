from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # Load .env variables
import os


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(MAIN_DIR)
CSV_PATH = os.path.join(MAIN_DIR, "data", "drugs_data.csv")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Using Neo4j credentials:")

df = pd.read_csv(CSV_PATH)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print("🔗 Connected to Neo4j database.")

def load_csv_to_neo(tx, row):
    q = """
    MERGE (a:Drug {name: $drug_a, smiles: $smiles_a, formula: $formula_a})
    MERGE (b:Drug {name: $drug_b, smiles: $smiles_b, formula: $formula_b})
    MERGE (a)-[r:INTERACTS_WITH {level: $level}]->(b)
    """
    tx.run(q,
           drug_a=row["Drug_A"], drug_b=row["Drug_B"], level=row["Level"],
           smiles_a=row["DrugA_SMILES"], formula_a=row["DrugA_Formula"],
           smiles_b=row["DrugB_SMILES"], formula_b=row["DrugB_Formula"])

with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(load_csv_to_neo, row)
print("✅ Graph data loaded into Neo4j.")

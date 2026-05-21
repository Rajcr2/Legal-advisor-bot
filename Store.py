import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def upload(pdf_input):
    def upload_single(pdf_path):
        pdf_name = os.path.basename(pdf_path).split(".pdf")[0]

        try:
            conn = psycopg2.connect(
                dbname= os.getenv("DB_NAME"),
                user= os.getenv("DB_USER"),
                password= os.getenv("DB_PASSWORD"),
                host= os.getenv("DB_HOST"),
                port= os.getenv("DB_PORT")
            )
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM legal_docs WHERE name = %s", (pdf_name,))
            if cursor.fetchone():
                print(f"⚠️ Skipping upload. '{pdf_name}' already exists.")
                cursor.close()
                conn.close()
                return

            with open(pdf_path, 'rb') as file:
                binary_data = file.read()

            insert_query = "INSERT INTO legal_docs (name, pdf_data) VALUES (%s, %s)"
            cursor.execute(insert_query, (pdf_name, psycopg2.Binary(binary_data)))
            conn.commit()

            print(f"✅ Uploaded '{pdf_name}' successfully.")
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error uploading '{pdf_path}':", e)

    if isinstance(pdf_input, str):
        upload_single(pdf_input)
    elif isinstance(pdf_input, list):
        print("📦 Starting batch upload...")
        for path in pdf_input:
            upload_single(path)
        print("✅ Batch upload completed.")
    else:
        print("❌ Invalid input. Please provide a file path or list of file paths.")

if __name__ == "__main__":

    directory = r"C:\Users\Raj\Desktop\ML_Internship\Legal Advisor\Bare Acts"
    pdf_files = ["The Hindu Marriage Act, 1955.pdf", 
                 "The Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.pdf"]
    upload([f"{directory}\\{pdf}" for pdf in pdf_files])


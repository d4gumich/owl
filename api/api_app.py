import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery

# Initialize app and dependencies
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
client = bigquery.Client()

# API Endpoint
@app.route('/similarity', methods=['POST'])
def similarity_search():
    try:
        # Parse request
        data = request.json
        input_text = data.get("text")
        k = int(data.get("k", 10))  # Default to 10 items if k is not provided

        if not input_text:
            return jsonify({"error": "Text input is required"}), 400

        # Generate input embedding
        input_embedding = model.encode(input_text).tolist()

        # BigQuery ANN search
        query = """
        SELECT 
            *,
            ML.DISTANCE(embedding, @input_embedding, 'COSINE') AS similarity
        FROM 
            `eternal-galaxy-447417-u8.humanitarian_db.pages_metadata`
        ORDER BY 
            similarity ASC
        LIMIT @k;
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("input_embedding", "FLOAT64", input_embedding),
                bigquery.ScalarQueryParameter("k", "INT64", k)
            ]
        )

        query_job = client.query(query, job_config=job_config)
        results = [{"uuid": row["uuid"], "id": row["id"], "page_label": row["page_label"], "title": row["title"],
                    "document": row["document"], "summary_page": row["summary_page"], "date_created": row["date_created"],
                    "year": row["year"], "disaster": row["disaster"],"feature": row["feature"],"file": row["file"],
                    "ocha_product": row["ocha_product"], "origin": row["origin"],"country_name": row["country_name"],
                    "source": row["source"],"theme_name": row["theme_name"],"URL": row["URL"],"combined_details": row["combined_details"],
                    "embedding": row["embedding"],"similarity": row["similarity"]} for row in query_job]


        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

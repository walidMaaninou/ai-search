import streamlit as st
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re
import json

# Load model and tokenizer only once
def load_model_and_tokenizer():
    model_name = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

def get_embedding(text):
    if not text:
        return [0.0] * 768
    text = normalize_arabic(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        mean_pooled = sum_embeddings / sum_mask
    return mean_pooled.squeeze().cpu()

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    # text = re.sub(r"ى", "ي", text)
    # text = re.sub(r"ؤ", "و", text)
    # text = re.sub(r"ئ", "ي", text)
    # text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text

# OpenSearch connection
client = OpenSearch(
    hosts=[{'host': 'search-stg-lexisearch-vector2-hauxwhenbjxtwujtjv6pw7ijoq.aos.us-east-1.on.aws', 'port': 443}],
    http_auth=('api-user', 'R@NkEDcT6.C_cCKqKWTu'),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)

INDEX_NAME = "lexical_resources"

def vector_search(query, size=50):
    query_vec = get_embedding(query).tolist()
    body = {
        "size": size,
        "query": {
            "knn": {
                "embedding_headword": {
                    "vector": query_vec,
                    "k": size
                }
            }
        }
    }
    response = client.search(index=INDEX_NAME, body=body)
    return response["hits"]["hits"]

def fetch_doc_by_id(doc_id):
    try:
        res = client.get(index=INDEX_NAME, id=doc_id)
        return res["_source"]
    except:
        return None

def remove_embedding_keys(hit):
    # Copy to avoid mutating original
    filtered = dict(hit)
    source = filtered.get("_source", {}).copy()
    
    # Remove embedding keys if present
    source.pop("embedding_headword", None)
    source.pop("embedding_definition", None)
    
    filtered["_source"] = source
    return filtered

def get_id_of_first_result(hit):
    entries = hit["_source"].get("entries", [])
    if entries:
        return entries[0].get("id"), entries[0].get("headword")  # Return ID and headword
    return None, None

def get_children_of_id(parent_id, size=50):
    parent_sense_ref = parent_id + "_sense"

    query_body = {
        "size": size,
        "query": {
            "nested": {
                "path": "relations",
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"relations.members.role.keyword": "part"}},
                            {"term": {"relations.members.ref.keyword": parent_sense_ref}}
                        ]
                    }
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query_body)
    return response['hits']['hits']

# Main app
st.title("Group-Based Semantic Search")

query = st.text_input("Enter Arabic headword:")
search_clicked = st.button("Search")

if search_clicked and query:
    with st.spinner("Searching..."):
        results = vector_search(query)

    if not results:
        st.warning("No results found.")
    else:
        filtered_results = [remove_embedding_keys(hit) for hit in results][:10]

        for i, hit in enumerate(filtered_results, start=1):
            source = hit["_source"]
            entries = source.get("entries", [])
            if not entries:
                continue
            entry = entries[0]
            headword = entry.get("headword", "N/A")
            senses = entry.get("senses", [])
            definition = senses[0]["definitions"][0].get("text", "N/A") if senses and senses[0].get("definitions") else "N/A"
            score = hit["_score"]

            st.markdown(f"**{i}. Headword:** {headword}")
            # st.markdown(f"    - Definition: {definition}")
            st.markdown(f"    - Score: {score:.4f}")
            st.markdown("---")


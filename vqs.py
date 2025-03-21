import json
import os
import pandas as pd
import csv
import pathlib
import textwrap
import io
import google.generativeai as genai

import streamlit as st
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from io import StringIO
from PIL import Image

from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AURA_INSTANCEID = os.getenv("AURA_INSTANCEID")
AURA_INSTANCENAME = os.getenv("AURA_INSTANCENAME")

huggingface_api_key = os.getenv("huggingface_api_key")
cohere_api_key = os.getenv("cohere_api_key")
langchain_api_key = os.getenv("langchain_api_key")
tavily_api_key = os.getenv("tavily_api_key")
gemini_api_key = os.getenv("gemini_api_key")
groq_api_key = os.getenv("groq_api_key")

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

graph = Neo4jGraph(
    url= NEO4J_URI,
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD
)

driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

genai.configure(api_key=gemini_api_key)

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

input_prompt = """
You are given an image that contains a pipeline of a software system.
You must extract information in CSV format with no headers:
Source_Module, Destination_Module, Information_Passed.
Do not include any extra characters.
"""
prompt = """
You are given an image which contains pipiline of a software, and you have give me the response in such way that it will have source module, 
destination module, and the information which is getting passed from one to another, in form of csv data and you do not need add header and 
any extra character
"""

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content([input,image[0],prompt])
    return response.text
    
# Function to insert data into Neo4j
def insert_data_into_neo4j(tx, data):
    for row in data:
        # Cypher query to create nodes and relationships
        cypher_query = """
        MERGE (source:Module {name: $Source_Module})
        MERGE (destination:Module {name: $Destination_Module})
        MERGE (info:Information {info: $Information_Passed})

        MERGE (source)-[:PASSES]->(destination)
        MERGE (source)-[:INCLUDES]->(info)
        """
        tx.run(cypher_query, 
               Source_Module=row['Source_Module'], 
               Destination_Module=row['Destination_Module'], 
               Information_Passed=row['Information_Passed'])
        
# Function to execute any Neo4j query
def execute_neo4j_query(query):
    with driver.session() as session:
        result = session.run(query)
        data = [dict(record) for record in result]
    
    if not data:
        return "⚠️ No results found.", None
    
    df = pd.DataFrame(data)
    return df.to_string(index=False), df
        
# Function to delete all nodes and relationships
def delete_all_nodes():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    return "✅ All nodes and relationships deleted successfully!"

def show_all_nodes():
    with driver.session() as session:
        nodes_result = session.run("MATCH (n) RETURN n")
        edges_result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")

        nodes = [record["n"] for record in nodes_result]
        edges = [(record["n"].element_id, record["m"].element_id) for record in edges_result]

    G = nx.Graph()

    # for node in nodes:
    #     properties = dict(node.items())
    #     G.add_node(node.element_id, label=list(node.labels), properties=properties)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # %matplotlib inline
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color="grey", font_size=10)
    plt.title("📊 Neo4j Graph Visualization")
    plt.show()

    image_path = "graph.png"
    plt.savefig(image_path)
    plt.close()
    
    return image_path

# Function to process images and text
def process_input(user_query, image_paths):
    all_extracted_data = []
    all_dfs = []

    if not image_paths:
        return "⚠️ No images provided.", None

    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_parts = [{"mime_type": "image/png", "data": image_bytes}]

        # Generate response using Gemini AI
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([user_query] + image_parts + [input_prompt])
        extracted_data = response.text  # RAW CSV TEXT

    # 🛑 DEBUG: Print the raw extracted text
        print(f"🔹 Extracted CSV from {image_path}:\n", extracted_data)

        all_extracted_data.append(f"📄 Image: {image_path}\n{extracted_data}\n\n")

        # Convert response to DataFrame
        try:
            csv_reader = StringIO(extracted_data)
            df = pd.read_csv(csv_reader, header=None, names=["Source_Module", "Destination_Module", "Information_Passed"])
            all_dfs.append(df)
        except Exception as e:
            all_extracted_data.append(f"⚠️ Error parsing CSV for {image_path}: {str(e)}\n")
            continue

    # Combine all extracted DataFrames
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = None
    
    # Insert data into Neo4j
    if final_df is not None:
        with driver.session() as session:
            session.execute_write(insert_data_into_neo4j, final_df.to_dict(orient="records"))

    return "\n".join(all_extracted_data), final_df  # Return raw text + DataFrame

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🔍 Software Architecture Query System")

    # Image + Query Processing
    gr.Markdown("## 📂 Upload Software Pipeline Images & Ask Queries")
    query_input = gr.Textbox(label="Enter Your Query")
    image_input = gr.Files(label="Upload Software Pipeline Images")
    process_button = gr.Button("🚀 Process")
    
    extracted_text_output = gr.Textbox(label="🔍 Extracted CSV Text (Debugging)", lines=10)
    structured_data_output = gr.Dataframe(headers=["Source_Module", "Destination_Module", "Information_Passed"])

    process_button.click(process_input, inputs=[query_input, image_input], outputs=[extracted_text_output, structured_data_output])

    # Neo4j Graph Controls
    gr.Markdown("## 📊 Neo4j Graph Operations")
    
    with gr.Row():
        btn_show = gr.Button("📊 Show All Nodes")
        btn_delete = gr.Button("🗑️ Delete All Nodes")

    graph_output = gr.Image(type="filepath")
    
    btn_show.click(show_all_nodes, inputs=[], outputs=graph_output)
    btn_delete.click(delete_all_nodes, inputs=[], outputs=graph_output)

    # Custom Query Execution
    gr.Markdown("## ⚡ Execute Custom Neo4j Query")
    cypher_query_input = gr.Textbox(label="Enter Neo4j Query", placeholder="MATCH (n) RETURN n LIMIT 5")
    query_button = gr.Button("🚀 Run Query")
    query_output_text = gr.Textbox(label="Query Output (Text)")
    query_output_df = gr.Dataframe()

    query_button.click(execute_neo4j_query, inputs=[cypher_query_input], outputs=[query_output_text, query_output_df])

#==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>==>

app.launch(server_name="0.0.0.0", server_port=7870, share=True)

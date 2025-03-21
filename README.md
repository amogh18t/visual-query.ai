# 🔍 Software Architecture Query System

## 📖 Overview
This project is a **Software Architecture Query System** that utilizes **Neo4j**, **Google Gemini AI**, and **Gradio** to extract structured data from software pipeline images and visualize/query the architecture.

## 🚀 Features
- **Extract Information from Software Architecture Diagrams**
- **Insert Structured Data into Neo4j Graph Database**
- **Visualize Software Architecture as a Graph**
- **Run Custom Cypher Queries on Neo4j**
- **Query anything regarding the relations of data (working on it..)**
- **Interactive UI with Gradio**

## 🛠️ Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/amogh18t/visual-query.ai.git
cd visual-query.ai
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables
Create a `.env` file and add your credentials:
```
NEO4J_URI=<your_neo4j_uri>
NEO4J_USERNAME=<your_neo4j_username>
NEO4J_PASSWORD=<your_neo4j_password>
AURA_INSTANCEID=<your_aura_instanceid>
AURA_INSTANCENAME=<your_aura_instancename>

gemini_api_key=<your_gemini_api_key>
```

### 4️⃣ Run the Application
```sh
python app.py
```
The app will launch on **`http://0.0.0.0:7860`**.

## 📂 How It Works
1. **Upload an Image** of a software architecture pipeline.
2. **Google Gemini AI** extracts structured data (source, destination, and passed information) in CSV format.
3. **Neo4j** stores the structured data as a graph.
4. **Visualize the graph** or execute custom **Cypher Queries** via Gradio UI.

## 🏗️ Architecture
- **Neo4j** - Graph database to store software architecture relationships.
- **Google Gemini AI** - Extracts structured data from images.
- **Gradio** - Interactive UI for user interaction.
- **Matplotlib & NetworkX** - Graph visualization.

## 📊 Example Queries
### Show all nodes:
```cypher
MATCH (n) RETURN n
```

### Delete all nodes:
```cypher
MATCH (n) DETACH DELETE n
```

### Find all modules passing information:
```cypher
MATCH (s:Module)-[:PASSES]->(d:Module) RETURN s, d
```

## 📸 Sample Graph Output
After processing an image, the software pipeline can be visualized as a graph.

![Graph Visualization](graph.png)

## 👨‍💻 Contributors
- **Your Name** (your.email@example.com)

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


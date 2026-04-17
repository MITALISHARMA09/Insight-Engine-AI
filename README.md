# InsightEngine AI

InsightEngine AI is an AI-powered data analysis platform designed to make dataset understanding easier for non-technical users. It allows users to upload structured data files, automatically profile and clean the data, generate dashboard-ready insights, and ask questions in natural language without writing code.

The system combines a FastAPI backend, a lightweight Vanilla JavaScript frontend, a modular data processing pipeline, a Retrieval-Augmented Generation (RAG) layer, and a multi-agent LLM workflow for analysis and explanation.

---

## Overview

Real-world datasets are often incomplete, inconsistent, and difficult to analyze directly. InsightEngine AI was built to reduce that effort by automating the complete analysis workflow.

The platform supports:

- Dataset upload and validation
- Automatic profiling and quality analysis
- AI-assisted and rule-based data cleaning
- Domain detection
- Auto-generated dashboards
- Natural language querying over datasets
- Query history and metadata tracking
- RAG-powered contextual retrieval
- Multi-agent code generation and execution

---

## Key Features

### 1. Dataset Upload and Validation
Users can upload CSV or Excel files through the frontend. The backend validates the file type and size, reads the dataset into memory, and initializes project metadata for further processing.

### 2. Data Profiling
The system performs an initial study of the dataset to identify:

- Missing values
- Duplicate rows
- Outliers
- Type inconsistencies
- Column-level statistics
- Data quality indicators

This profiling stage helps the platform understand the condition of the dataset before cleaning begins.

### 3. Domain Detection
A dedicated domain expert module analyzes the column names, data types, and sample records to identify the likely domain of the dataset, such as sales, retail, HR, finance, or education. This helps later modules make more informed decisions.

### 4. AI-Assisted Data Cleaning
The cleaning workflow combines AI-generated planning with rule-based safeguards. Based on the profiling report and detected domain, the system proposes and applies cleaning operations such as:

- Handling missing values
- Fixing data types
- Removing duplicates
- Managing outliers
- Standardizing categories

### 5. Auto Dashboard Generation
After cleaning, the platform generates an automatic dashboard using dataset structure and statistics. This may include:

- KPI cards
- Category breakdowns
- Distributions
- Time-based trends
- Correlation-based views
- Missing-value summaries

### 6. Natural Language Querying
Users can ask questions about the dataset in plain English. The system converts the query into executable pandas logic using a multi-agent LLM workflow, executes it in a controlled environment, and returns a human-readable answer.

### 7. Retrieval-Augmented Generation (RAG)
Before answering user queries, the system retrieves the most relevant dataset context from prebuilt chunks and embeddings. This improves relevance and reduces hallucination during query generation.

### 8. Query History and Metadata Storage
SQLite is used to track:

- Uploaded dataset metadata
- Cleaning reports
- Profiling reports
- Query history
- Execution status
- Errors and timestamps

---

## Project Architecture

The project follows a modular structure to keep responsibilities separated and maintainable.

### Backend
The backend is built with **FastAPI** and **Python**. It handles:

- API routing
- File upload and processing
- Cleaning orchestration
- Dashboard generation
- Query execution
- Metadata persistence

### Frontend
The frontend is built using **Vanilla JavaScript**, **HTML**, and **CSS** inside a lightweight single-page interface. It provides screens for:

- Uploading datasets
- Viewing dashboards
- Previewing data
- Asking questions
- Reviewing query history

### Database
**SQLite** is used as the local persistence layer for dataset metadata and user query history.

### RAG Layer
The RAG module converts dataset summaries into text chunks, transforms them into embeddings, and stores them in an index for relevant retrieval during question answering.

### Multi-Agent LLM Workflow
The system uses multiple LLM-powered modules to improve answer quality and robustness.

---

## Module-Wise Workflow

### 1. Application Startup
The application starts from `main.py`, which:

- Initializes the FastAPI app
- Sets up middleware
- Loads the frontend
- Creates the database tables
- Registers API routes

### 2. File Upload
The user uploads a CSV or Excel file from the frontend. The upload endpoint reads the file, validates it, generates a dataset identifier, and stores metadata.

### 3. Profiling and Domain Analysis
The uploaded dataset is analyzed for structure and quality. At the same time, the domain expert identifies the likely dataset domain.

### 4. Cleaning Plan Generation
An AI cleaner creates a structured cleaning plan using the profiling report and domain information.

### 5. Cleaning Execution
The cleaning executor applies the plan safely using domain rules and rule-based constraints.

### 6. RAG Index Build
The cleaned dataset is converted into structured text chunks. These are embedded using a sentence-transformer model and stored in an index for retrieval.

### 7. Dashboard Generation
An automatic dashboard is generated using the processed dataset and statistical summaries.

### 8. Query Processing
When the user asks a question, the system:
- retrieves relevant dataset context from RAG
- sends the context and schema to two coder agents
- evaluates both code outputs using a judge agent
- executes the selected code in a sandbox
- converts the result into a plain-English explanation

### 9. History Storage
The final query, code, execution result, and response summary are stored in SQLite for future reference.

---

## LLM Agents Used

### Domain Expert
Identifies the dataset domain using schema, data types, and sample rows.

### Coder A
Generates pandas-based analytical code from the user’s question and dataset context.

### Coder B
Generates an alternative code solution using a separate model/provider path.

### Judge
Compares outputs from both coder agents, selects the stronger solution, and may also help repair failed code.

### Storyteller / Insight Narrator
Converts technical outputs into clear, human-readable explanations for users.

### Rule-Based Coder
Provides a fallback logic path when LLM-generated code is unavailable or unreliable.

### Cleaning Executor
Applies approved data cleaning steps to the dataset in a controlled and traceable way.

### Auto Dashboard Engine
Generates visual summaries and key metrics directly from the dataset without manual chart configuration.

---

## Technologies Used

### Python
Used for backend development, orchestration, data analysis, cleaning, RAG, and AI integration.

### FastAPI
Used to build the backend API and connect the frontend with processing modules.

### Vanilla JavaScript
Used to create the lightweight frontend interface and communicate with backend endpoints.

### SQLite
Used for local storage of dataset metadata, reports, and query history.

### Pandas / NumPy
Used for data loading, cleaning, transformation, analysis, and summary generation.

### Sentence Transformers
Used to create embeddings from dataset chunks for semantic retrieval.

### FAISS
Used as the vector index when available for efficient similarity search over chunk embeddings.

### Groq
Used for fast LLM inference in multiple AI agents.

### OpenRouter
Used to support alternative model access, especially for the second coder path.

---

## LLM Models Used

The exact model configuration may vary based on environment setup, but the project is designed around models such as:

- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `google/gemma-4-26b-a4b-it:free`

### Brief Purpose of These Models

- **Llama 3.3 70B Versatile**  
  Used for stronger reasoning, code generation, and comparison tasks where higher quality output is important.

- **Llama 3.1 8B Instant**  
  Used for faster lightweight tasks such as domain understanding, cleaner support, or explanation flows.

- **Gemma-based model**  
  Used as an alternative model path depending on availability and configuration.

---

## Why SQLite, RAG, and Multi-Agent Design Were Used

### Why SQLite
SQLite provides a simple and lightweight local database with no separate server requirement. It is suitable for storing dataset metadata, query logs, and execution records in a compact full-stack project.

### Why RAG
RAG helps the system retrieve relevant dataset context before generating code or answers. This improves relevance and makes responses more grounded in the uploaded data.

### Why Multi-Agent LLM Design
Instead of relying on a single LLM response, the system uses multiple specialized roles. This improves modularity, robustness, and answer quality by separating responsibilities such as understanding, coding, judging, and explaining.

---

## Folder Structure

```text
InsightEngine/
│
├── app/
│   ├── agents/
│   ├── analysis/
│   ├── api/
│   ├── cleaning/
│   ├── db/
│   ├── engine/
│   ├── rag/
│   
│
├── frontend/
│   └── index.html
│
├── main.py
```


## Current Limitations
- Frontend is lightweight and may need expansion for larger-scale production use
- LLM quality depends on model/provider availability
- Best suited for structured tabular datasets such as CSV and Excel
- Large-scale enterprise datasets may require further optimization
- Query accuracy depends on schema clarity and generated code quality

## Use Cases

InsightEngine AI is suitable for:

- Academic and internship projects
- Quick dataset understanding
- Business data exploration
- Automated preliminary analytics
- Demonstrating AI-assisted data workflows
- Helping non-technical users interact with structured data

## Future Scope

Possible future improvements include:

- More advanced frontend components
- Better visualization interactivity
- Stronger chart rendering in chat responses
- More robust data validation rules
- Expanded support for larger datasets
- Improved agent coordination and response reliability
- Enhanced deployment and authentication support

## Conclusion

InsightEngine AI demonstrates a practical approach to AI-assisted data analysis by combining data engineering, intelligent cleaning, automated dashboarding, semantic retrieval, and multi-agent reasoning in a single modular platform.

It was conceptualized as a structured system where each module has a clear responsibility, making the project both technically meaningful and extensible for future work.

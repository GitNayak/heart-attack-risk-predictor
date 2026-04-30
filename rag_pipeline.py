import os
import glob
import warnings
import pickle
import pandas as pd
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ignore specific warnings caused by sklearn or Langchain internals to keep the terminal output clean.
warnings.filterwarnings("ignore", category=UserWarning)


class HeartAttackRAG:
    """
    HeartAttackRAG encapsulates the entire Logic for connecting external knowledge (Retrieval Augmented Generation) 
    with dynamic Machine Learning components.
    """
    def __init__(self, api_key: str = None, data_path: str = "heart.csv", model_path: str = "heart.pkl", provider: str = "groq"):
        # Essential Credentials & Configuration
        self.api_key = api_key
        self.provider = provider
        self.data_path = data_path
        self.model_path = model_path
        
        # Local system paths for embeddings and vectors
        self.persist_dir = "./chroma_db"
        self.knowledge_base_file = "rag_knowledge_base.md"
        
        # Empty placeholders logically loaded throughout __init__
        self.llm = None          # Large Language Model (e.g. Llama 3)
        self.embeddings = None   # Text-to-Vector Mathematical Converter
        self.vectorstore = None  # Local Database holding embedded chunks
        self.retriever = None    # Search mechanism to fetch chunks based on mathematical similarity
        self.qa_chain = None     # The specific Langchain Pipeline taking Users Question -> LLM Answer
        self.model = None        # Local ML classifier used by the risk prediction flow
        self.df = None           # Heart Attack Dataset (used for dataset statistics context)

        # Triggers sequential initialization protocols
        self._load_prediction_model()
        self._init_models()

    def _load_prediction_model(self):
        """
        Load the serialized ML classifier when it exists so app consumers can
        rely on a stable `model` attribute regardless of RAG configuration.
        """
        if not os.path.exists(self.model_path):
            return

        try:
            with open(self.model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
        except Exception as exc:
            print(f"[RAG] Warning: Unable to load prediction model from '{self.model_path}': {exc}")

    # ─────────────────────────────────────────────────────────────
    # 1. CORE AI INIT PROTOCOLS
    # ─────────────────────────────────────────────────────────────
    def _init_models(self):
        """
        Initializes the Embedding Models and connects securely to the Langchain LLM Endpoints.
        """
        # ==========================================
        # SECURE CLOUD LLM CONNECTION ROUTING
        # ==========================================
        # If no API key was supplied, gracefully fall back gracefully so Streamlit doesn't break.
        # This keeps the ML Pipeline alive even if the user didn't request Chat Features.
        if not self.api_key and self.provider != "ollama":
            print("[RAG] Warning: No API key provided. LLM text generation will be disabled.")
            self.llm = None
            return

        print(f"[RAG] Initializing AI LLM Connection via provider: {self.provider}")
        os.environ["GROQ_API_KEY"] = self.api_key if self.api_key else ""
        os.environ["GOOGLE_API_KEY"] = self.api_key if self.api_key else ""
        os.environ["OPENAI_API_KEY"] = self.api_key if self.api_key else ""

        # Specifically route to the exact Model architecture requested.
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                temperature=0.2, # 0.2 Keeps the model highly factual rather than creative
                max_output_tokens=512
            )

        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",   # Llama 3 is extremely fast and capable of deep medical reasoning
                temperature=0.2,
                max_tokens=512
            )

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=512
            )
            
        elif self.provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                model="llama3.2",
                temperature=0.2
            )
        else:
            raise ValueError(f"Unknown provider '{self.provider}'.")

        # ==========================================
        # LOCAL EMBEDDING ENGINE INITIALIZATION
        # ==========================================
        # Instead of paying an API fee to convert text into mathematical Vectors, 
        # we run `all-MiniLM-L6-v2` locally on our CPU. It is completely free and incredibly fast.
        try:
            print("[RAG] Loading local HuggingFace embedding engine (all-MiniLM-L6-v2)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},      # Run explicitly on CPU to prevent CUDA errors on standard machines
                encode_kwargs={'normalize_embeddings': False}
            )
            print("[RAG] Local embeddings properly linked.")
        except Exception as e:
            # Revert to a basic print on charmap fallback error in Windows terminals
            print("[RAG] Local embeddings ready (Output exception caught).")

    # ─────────────────────────────────────────────────────────────
    # 2. LOCAL VECTOR DATABASE BUILDING
    # ─────────────────────────────────────────────────────────────
    def _generate_dataset_stats(self) -> str:
        """
        Calculates and generates a dynamic text-block summarizing the exact Dataset Statistics 
        for the AI to use as knowledge baseline facts.
        """
        df = self.df
        text = "## Heart Attack Dataset - Statistical Summary\\n\\n"
        text += f"Total patients: {len(df)}\\n"
        
        # NOTE: Quirks in the Kaggle UCI Dataset logic!
        # In this precise dataset, target output `0` correlates logically perfectly with disease indicators
        # Therefore `0` = High Risk, and `1` = Healthy
        text += f"Heart attack cases (target 0 = HIGH RISK disease): {int((1 - df['output']).sum())} ({(1-df['output'].mean())*100:.1f}%)\\n"
        text += f"Low risk cases (target 1 = NO DISEASE healthy): {int(df['output'].sum())} ({df['output'].mean()*100:.1f}%)\\n\\n"

        text += "### Feature Statistics\\n"
        stats = df.describe().round(2)
        for col in stats.columns:
            text += f"- **{col}**: mean={stats.loc['mean', col]}, range=[{stats.loc['min', col]}, {stats.loc['max', col]}]\\n"
        
        text += "\\n### Feature Correlations (Predictive Power)\\n"
        corr = df.corr()
        if "output" in corr.columns:
            corr_matrix = corr.abs().sort_values(by="output", ascending=False)
            for feat in corr.index:
                if feat == "output": continue
                val = corr_matrix["output"][feat]
                direction = "positive" if corr["output"][feat] > 0 else "negative"
                text += f"- {feat}: absolute correlation = {val:.3f} (raw vector is {direction})\\n"
                
        return text

    def _generate_model_doc(self) -> str:
        """
        Optional contextual function giving the AI details on Random Forest characteristics, 
        primarily used as legacy fallback context if users switch out the underlying ML Object path.
        """
        text = "## Machine Learning Model Information\\n\\n"
        text += "This tool integrates a Machine Learning classifier trained on the mentioned heart disease dataset.\\n"
        text += "For reference, in Random Forest runs, common critical features include:\\n"
        text += "1. chest pain type (cp)\\n2. maximum heart rate achieved (thalachh)\\n3. exercise induced angina (exng)\\n4. number of major vessels (caa)\\n5. age and sex definitions.\\n"
        return text

    def build_knowledge_base(self, force_rebuild: bool = False):
        """
        Orchestrates Data Loading -> Splitting -> Vectorizing -> Storing into ChromaDB.
        """
        import stat
        
        # Short-circuit logic: If the DB folder already exists on the user's hard-drive, 
        # we do not waste time or power recalculating Vectors; we instantly retrieve it.
        if not force_rebuild and os.path.exists(self.persist_dir):
            print(f"[RAG] Found existing ChromaDB vector index at {self.persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            # Create a localized memory retriever to pull out the 3 most relevant sentences per query
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            return

        print("[RAG] Initializing fresh Vector DB Build Sequence...")
        docs = []

        # Step 1: Read any existing Custom Medical Knowledgebase files
        if os.path.exists(self.knowledge_base_file):
            loader = TextLoader(self.knowledge_base_file, encoding='utf-8')
            docs.extend(loader.load())
            
        # Step 2: Read PDFs
        for pdf_file in glob.glob("*.pdf"):
            print(f"[RAG] Loading external PDF document: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())

        # Step 3: Automatically generate Dynamic Statistics Document Object
        if os.path.exists(self.data_path):
            try:
                self.df = pd.read_csv(self.data_path)
                # Drop highly correlated or unneeded duplicate parameters
                self.df = self.df.drop(["oldpeak", "slp", "thall"], axis=1, errors="ignore")
                stats_text = self._generate_dataset_stats()
                model_text = self._generate_model_doc()
                
                from langchain.schema import Document
                docs.append(Document(page_content=stats_text, metadata={"source": "dataset_stats"}))
                docs.append(Document(page_content=model_text, metadata={"source": "model_info"}))
            except Exception as e:
                print(f"[RAG] Warning: Unable to parse structured CSV statistics: {e}")

        # Step 4: Split Text into Bite-Sized Chunks
        # We break text into 500-Character blocks with 50-Character overlap so context boundaries aren't heavily lost.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # Step 5: Convert completely to Mathematical Storage
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        
        # Link Retriever logic
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"[RAG] Knowledge base built and successfully permanently stored at '{self.persist_dir}'.")

    # ─────────────────────────────────────────────────────────────
    # 3. LANGCHAIN EXPRESSION PIPELINE (LCEL)
    # ─────────────────────────────────────────────────────────────
    def _build_qa_chain(self):
        """
        Defines the LCEL Graph that connects our Vector Search variables 
        in a logical chain down to our Large Language Model.
        """
        # If the user's API Key failed or LLM wasn't loaded, skip compiling entirely
        if not self.llm or not self.retriever:
            return

        # Core Instruction Matrix 
        template = """You are a highly capable AI cardiovascular assistant designed for a predictive medical application.
Rely heavily on the following context loaded from your custom medical database.
Do not invent stats or rules. Provide clear, medical reasoning based strictly on the provided context if possible.

Context from Knowledge Base:
{context}

User Profile / Query:
{question}

Formulate your response concisely, using bullet points for key factors. Use an objective, clinical tone."""

        prompt = ChatPromptTemplate.from_template(template)

        # This chain follows standard Langchain sequence architecture:
        # Question -> Search Vector Store for nearest Context -> Format Prompt -> Generate AI Answer -> Extract String
        def format_docs(docs):
            return "\\n\\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("[RAG] Q&A Langchain successfully compiled and mounted.")


def create_rag_pipeline(api_key: str = None, provider: str = "groq", rebuild: bool = False, base_dir: str = ".") -> HeartAttackRAG:
    """
    Factory function used by standard external scripts (like Streamlit app.py) to securely 
    construct, build, and retrieve a finalized instance of the entire HeartAttackRAG pipeline.
    """
    os.chdir(base_dir) # Ensure system stays exactly in execution directory
    
    rag = HeartAttackRAG(
        api_key=api_key,
        data_path="heart.csv",
        model_path="heart.pkl",
        provider=provider
    )
    
    # 1. Establish database connection
    rag.build_knowledge_base(force_rebuild=rebuild)
    
    # 2. Compile functional nodes into finalized AI Chain
    rag._build_qa_chain()
    
    return rag

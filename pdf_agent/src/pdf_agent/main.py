#!/usr/bin/env python
import sys
import warnings
import requests
from io import BytesIO
from datetime import datetime
from pypdf import PdfReader
from composio_langchain import Action, App, ComposioToolSet
from pdf_agent.crew import PdfAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Initialize Composio ToolSet
composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.RAGTOOL])

def extract_text_from_pdf_url(pdf_url):
    """
    Download and extract text from a PDF file given its URL.
    Uses a User-Agent header to prevent request rejection.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/pdf"
    }

    try:
        # Download the PDF file with headers
        response = requests.get(pdf_url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Read the PDF content
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text.strip():
            raise ValueError("No extractable text found in the PDF. It may be an image-based PDF.")

        return text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error while fetching PDF: {e}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")
    

def run():
    """
    Run the crew.
    """
    pdf_url = str(input("Enter the PDF URL: "))
    print("Extracting PDF content from URL...")

    try:
        # Extract text from the PDF
        pdf_content = extract_text_from_pdf_url(pdf_url)

        # Prepare inputs for the crew
        inputs = {
            'pdf_content': pdf_content  # Pass the extracted PDF content
        }

        # Run the crew
        PdfAgent().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    pdf_content = extract_text_from_pdf_url(pdf_url)
    inputs = {
            'pdf_content': pdf_content  # Pass the extracted PDF content
        }
    try:
        PdfAgent().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PdfAgent().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        PdfAgent().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    # Example: python main.py run
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "run":
            run()
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        else:
            print("Invalid command. Use 'run', 'train', 'replay', or 'test'.")
    else:
        print("Please provide a command: 'run', 'train', 'replay', or 'test'.")
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:/vislona/chatbot/dataset/chatbot data.pdf")  # <-- Specify your PDF file path here
pages_content = loader.load_and_split()
print(len(pages_content), pages_content)

import os

# Define folder structure
folders = [
    "App",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "src/pipeline"
    "scripts"
    "streamlit_app",
    "data/raw",
    "data/processed",
    
]

def generate_project():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Project structure and boilerplate scripts created.")

if __name__ == "__main__":
    generate_project()
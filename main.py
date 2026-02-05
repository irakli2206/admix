from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess

app = FastAPI()

# Allow your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your Vercel URL later for security
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Kvali Engine is running"}


@app.post("/raw-to-k36")
async def process_dna(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = subprocess.run(
            ["admix", "-f", temp_path, "-v", "23andme", "-m", "K36"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse output into JSON
        clean_results = {}
        for line in result.stdout.split("\n"):
            if ":" in line and "%" in line:
                name, value = line.split(":")
                clean_results[name.strip()] = float(value.replace("%", "").strip())

        return {"status": "success", "results": clean_results}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": e.stderr}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ML-based-IDS
A machine learning-based intrusion detection system that analyzes network traffic to detect cyber attacks in real-time using XGBoost, achieving 92.9% recall and 90.3% precision on the UNSW-NB15 dataset.

Before you begin, ensure the following software is installed on your machine:

| Software |	Version |	Purpose |
|----------|----------|---------|
| Python |	3.10 or higher |	Backend runtime |
| Node.js |	18 or higher |	Frontend runtime |
| Git	| Latest |	Cloning the repository |
| pip	| Latest |	Python package manager |
| npm |	Latest |	Node.js package manager |

<u>Downloading from GitHub</u>
Step 1: Clone the Repository
Open a terminal (Command Prompt, PowerShell, or Git Bash) and run:
git clone https://github.com/mutio18/aegis-ids.git

Step 2: Navigate into the Project Directory
bash: cd aegis-ids
You will see two main folders:
- ml-backend/ - FastAPI backend with XGBoost model
- aegis-frontend/ - Next.js frontend application

<u>Backend Setup</u>
Step 1: Navigate to Backend Folder
bash: cd ml-backend

Step 2: Create Virtual Environment
Windows:
bash:
python -m venv venv
venv\Scripts\activate

Mac / Linux:
bash
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
bash:
pip install -r requirements.txt
This installs FastAPI, Uvicorn, XGBoost, scikit-learn, pandas, numpy, SQLAlchemy, and other required packages.

Step 4: Verify the Model File
Ensure the trained XGBoost model exists in the models/ folder:
bash:
dir models\        # Windows
ls models/         # Mac/Linux
You should see a file named xgboost_mixed.pkl. If it is missing, download it from the repository.

Step 5: Start the Backend Server
bash: uvicorn main:app --reload --port 8000
You will see output similar to:
text
Model loaded with 18 features
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Keep this terminal window open. The backend must remain running for the frontend to work.

<u>Frontend Setup</u>
Step 1: Open a New Terminal
Open a second terminal window while keeping the backend terminal running.

Step 2: Navigate to Frontend Folder
bash: cd aegis-frontend
Step 3: Install Dependencies
npm install
This installs Next.js, React, TypeScript, Tailwind CSS, Recharts, and other required packages.

Step 4: Start the Frontend Development Server
bash: npm run dev
You will see output similar to:
text
▲ Next.js 14.x.x
- Local:        http://localhost:3000
- Ready in 2.3s

Running the System
Step 1: Access the Application
Open your web browser and navigate to:
text
http://localhost:3000

Step 2: Register or Log In
New users: Click "Register" and create an account
Existing users: Log in with your credentials

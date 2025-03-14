# AI Face Detection Backend Setup

## Prerequisites
- Python installed (Recommended: Python 3.8+)
- PostgreSQL database set up
- Virtual environment (optional but recommended)

## Step 1: Clone the Repository
```sh
git clone <repository-url>
cd AI_Face_detection
```

## Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```sh
python -m venv .venv  # Create virtual environment
source .venv/bin/activate  # Activate on macOS/Linux
.venv\Scripts\activate  # Activate on Windows
```

## Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## Step 4: Set Up Environment Variables
Create a `.env` file in the project root and add the following:
```
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=your_database_port
```

## Step 5: Run the Backend Server
```sh
python app.py
```
By default, the server runs on `http://127.0.0.1:5001`.

## Step 6: Test the API
### Upload an Image for Face Matching
```sh
curl -X POST http://127.0.0.1:5001/match -F "image=@/path/to/image.jpg"
```
If a match is found, you’ll get a response with the person's details.

## Step 7: Stop the Server
Press `Ctrl + C` in the terminal.

---
Now you’re ready to use the AI Face Detection backend! 🚀


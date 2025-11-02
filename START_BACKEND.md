# 🚀 How to Start the Backend

## Quick Start

### **Option 1: From Project Root**
```bash
cd C:\Users\kanishka\signature-verifier
python -m backend.app
```

### **Option 2: From Backend Directory**
```bash
cd C:\Users\kanishka\signature-verifier\backend
python -m backend.app
```

### **Option 3: With Virtual Environment**
```bash
cd C:\Users\kanishka\signature-verifier
.venv\Scripts\activate
python -m backend.app
```

## ✅ **What You Should See:**

```
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

## 🔍 **Verify It's Running:**

Open a new terminal and run:
```bash
curl http://127.0.0.1:5000/health
```

Or visit in browser:
```
http://127.0.0.1:5000/health
```

Expected response:
```json
{
  "security": {
    "file_validation": true,
    "images_stored": false,
    "rate_limiting": true
  },
  "status": "ok"
}
```

## 🌐 **Frontend Connection:**

Once backend is running:
1. Start frontend: `cd frontend-vite && npm run dev`
2. Frontend will connect to: `http://127.0.0.1:5000`
3. Verify in browser: Open `http://localhost:8080` (or whatever port Vite shows)

## 🛑 **To Stop:**

Press `CTRL+C` in the terminal where Flask is running.


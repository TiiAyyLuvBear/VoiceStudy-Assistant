# Secure Voice Assistant frontend

React + TypeScript frontend for VoiceStudy. Uses real FastAPI endpoints; no mock command path. Microphone recordings convert to WAV in-browser before upload.

```powershell
cd frontend
npm install
npm run test:run
npm run build
npm run dev
```

Backend contract: `GET /health`, `POST /api/v1/process`, `POST /api/v1/enroll`, `GET /api/v1/users`, `DELETE /api/v1/users/{user_id}`. Start backend with `..\\.venv\\Scripts\\python.exe -m backend.main`. Raw audio is never stored in localStorage. CORS allows Vite origins `localhost:5173` and `127.0.0.1:5173`.

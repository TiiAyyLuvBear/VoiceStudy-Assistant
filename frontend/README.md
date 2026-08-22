# Secure Voice Assistant frontend

React + TypeScript frontend for VoiceStudy. Uses real FastAPI endpoints; no mock command path. Microphone recordings convert to WAV in-browser before upload.
Enrollment requires five prompted WAV samples and a 3-word secret phrase.
Assistant page shows the fixed command catalog, including slot templates for
general schedules and private notes. It records protected commands first, then
asks for the registered secret phrase as a second audio sample. Completed
responses are shown in the output panel and spoken through backend `POST
/api/v1/tts`.

```powershell
cd frontend
npm install
npm run test:run
npm run build
npm run dev
```

Backend contract: `GET /health`, `GET /api/v1/commands`, `POST
/api/v1/process`, `POST /api/v1/tts`, `POST /api/v1/enroll`, `GET
/api/v1/users`, `DELETE /api/v1/users/{user_id}`. Start backend with
`..\\.venv\\Scripts\\python.exe -m backend.main`. Raw audio is never stored in
localStorage. CORS allows Vite origins `localhost:5173` and `127.0.0.1:5173`.

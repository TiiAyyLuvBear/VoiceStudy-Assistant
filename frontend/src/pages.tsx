import React, {ChangeEvent, FormEvent, useEffect, useRef, useState} from 'react';
import {Link, useParams} from 'react-router-dom';
import {Check, LockKeyhole, Mic, RefreshCw, Trash2, Upload, UserPlus, Volume2} from 'lucide-react';
import {Badge, Card, Status} from './components';
import {
  deleteSpeaker,
  enrollSpeaker,
  ENROLLMENT_PROMPTS,
  getCommands,
  getSpeakers,
  processAudio,
  synthesizeSpeech,
  SYSTEM_COMMANDS,
} from './services/api';
import {useMediaRecorder} from './hooks/useMediaRecorder';
import type {AssistantResult, Speaker, SystemCommand, VerificationStatus} from './types';

const SHOW_RAW_TRANSCRIPT = import.meta.env.DEV || import.meta.env.VITE_SHOW_RAW_TRANSCRIPT === 'true';
const ASSISTANT_HISTORY_KEY = 'voicestudy.assistant.history.v1';
const MAX_ASSISTANT_HISTORY = 8;

type AssistantHistoryItem = {
  id: string;
  command: string;
  rawTranscript: string;
  response: string;
  result: AssistantResult;
  createdAt: string;
};

function loadAssistantHistory(): AssistantHistoryItem[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(ASSISTANT_HISTORY_KEY) || '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(item => item && typeof item.id === 'string' && typeof item.command === 'string' && typeof item.response === 'string')
      .slice(0, MAX_ASSISTANT_HISTORY);
  } catch {
    return [];
  }
}

function saveAssistantHistory(items: AssistantHistoryItem[]) {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(ASSISTANT_HISTORY_KEY, JSON.stringify(items));
}

function historyItemFromResult(result: AssistantResult): AssistantHistoryItem {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    command: result.commandText || result.transcript || 'Command not recognized',
    rawTranscript: result.transcript || '',
    response: result.response || result.error || 'No backend response returned.',
    result,
    createdAt: new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
  };
}

function LoadingStep({label, detail}: {label: string; detail?: string}) {
  return (
    <div className="work-indicator" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <div>
        <strong>{label}</strong>
        {detail && <small>{detail}</small>}
      </div>
      <span className="progress-line" aria-hidden="true" />
    </div>
  );
}

function LoadingBadge({label}: {label: string}) {
  return (
    <span className="loading-badge">
      <span className="mini-spinner" aria-hidden="true" />
      {label}
    </span>
  );
}

export function AssistantPage() {
  const recorder = useMediaRecorder();
  const [result, setResult] = useState<AssistantResult>();
  const [autoSpeechId, setAutoSpeechId] = useState('');
  const [commandAudio, setCommandAudio] = useState<Blob>();
  const [commands, setCommands] = useState<SystemCommand[]>(SYSTEM_COMMANDS);
  const [commandsLoading, setCommandsLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState('');
  const [error, setError] = useState('');
  const [verification, setVerification] = useState<VerificationStatus>('not-required');
  const [history, setHistory] = useState<AssistantHistoryItem[]>(loadAssistantHistory);

  useEffect(() => {
    setCommandsLoading(true);
    getCommands()
      .then(setCommands)
      .catch(() => setCommands(SYSTEM_COMMANDS))
      .finally(() => setCommandsLoading(false));
  }, []);

  const needsSecret = (next: AssistantResult) =>
    next.securityLevel === 'verified' &&
    ['SECRET_PHRASE_REQUIRED', 'SECRET_PHRASE_FAILED', 'SECRET_PHRASE_NOT_CONFIGURED', 'SECRET_PHRASE_ASR_FAILED'].includes(next.error || '');
  const isSupportedAudio = (file: Blob & {name?: string}) =>
    file.type === 'audio/wav' || file.type === 'audio/flac' || !!file.name?.toLowerCase().match(/\.(wav|flac)$/);

  const runAudio = async (audio: Blob, secretAudio?: Blob) => {
    setError('');
    setBusy(true);
    setBusyLabel(secretAudio ? 'Verifying secret phrase' : 'Processing command audio');
    if (!secretAudio) setResult(undefined);
    try {
      const next = await processAudio(audio, secretAudio);
      setResult(next);
      if (needsSecret(next) && !secretAudio) {
        setCommandAudio(audio);
        setVerification('required');
      } else {
        setVerification(next.securityLevel === 'verified' ? (next.verified ? 'verified' : 'rejected') : 'not-required');
        const historyItem = historyItemFromResult(next);
        setAutoSpeechId(historyItem.id);
        setHistory(current => {
          const updated = [historyItem, ...current].slice(0, MAX_ASSISTANT_HISTORY);
          saveAssistantHistory(updated);
          return updated;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Audio/API error');
      setVerification('error');
    } finally {
      setBusy(false);
      setBusyLabel('');
    }
  };

  const toggle = async () => {
    setError('');
    try {
      if (recorder.recording) await runAudio(await recorder.stop());
      else await recorder.start();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Microphone error');
    }
  };

  const verifySecret = async () => {
    setError('');
    if (!commandAudio) {
      setVerification('error');
      setError('Command audio missing. Record command again.');
      return;
    }
    try {
      if (recorder.recording) {
        setVerification('verifying');
        await runAudio(commandAudio, await recorder.stop());
      } else {
        await recorder.start();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Microphone error');
      setVerification('error');
    }
  };

  const upload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!isSupportedAudio(file)) {
      setError('Only WAV or FLAC audio is accepted.');
      return;
    }
    void runAudio(file);
  };

  const uploadSecret = (event: ChangeEvent<HTMLInputElement>) => {
    setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    if (!commandAudio) {
      setVerification('error');
      setError('Command audio missing. Record command again.');
      return;
    }
    if (!isSupportedAudio(file)) {
      setError('Only WAV or FLAC audio is accepted.');
      return;
    }
    setVerification('verifying');
    void runAudio(commandAudio, file);
  };

  const heroStatus = recorder.recording
    ? verification === 'required' || verification === 'verifying'
      ? 'Reading secret phrase... click microphone to stop'
      : 'Listening... click microphone to stop'
    : busy
      ? `${busyLabel}...`
      : verification === 'required'
        ? 'Read your registered secret phrase now.'
        : 'Microphone ready';

  return (
    <div className="assistant-stack" aria-busy={busy}>
      <div className={`hero assistant-hero ${busy ? 'is-working' : ''}`}>
        <span className="hero-kicker">FIXED COMMAND PIPELINE</span>
        <h2>Speak a supported command.</h2>
        <p>Use one fixed phrase below so ASR post-processing can snap noisy text to system intent.</p>
        {commandsLoading ? (
          <LoadingStep label="Loading command catalog" detail="Preparing fixed phrases" />
        ) : (
          <div className="command-list">
            {commands.map(command => (
              <button className="command-chip" key={command.phrase} type="button">
                {command.phrase}
                {command.requires_secret && <LockKeyhole size={13} />}
              </button>
            ))}
          </div>
        )}
        <div className="audio-actions">
          <button
            className={`mic ${recorder.recording || busy ? 'busy' : ''}`}
            type="button"
            onClick={verification === 'required' || verification === 'verifying' ? verifySecret : toggle}
            aria-label={recorder.recording ? 'Stop recording' : 'Start recording'}
            disabled={busy && !recorder.recording}
          >
            <Mic size={32} />
          </button>
          <label className={`button secondary upload-button ${busy ? 'disabled' : ''}`}>
            <Upload size={17} /> Upload WAV/FLAC
            <input type="file" accept="audio/wav,audio/flac,.wav,.flac" onChange={upload} disabled={busy} />
          </label>
        </div>
        <p>{heroStatus}</p>
        {busy && <LoadingStep label={busyLabel || 'Processing'} detail="Backend ASR, speaker check, and command routing are running" />}
        {error && <Status label={error} tone="danger" />}
      </div>

      {result?.securityLevel === 'verified' && verification !== 'verified' && (
        <SecurityGate
          status={verification}
          onVerify={verifySecret}
          onUploadSecret={uploadSecret}
          recording={recorder.recording}
          secretTranscript={result.secretPhraseTranscript}
        />
      )}

      <section className="qa-history" aria-label="Assistant history">
        <div className="section-title qa-history-title">
          <span>Command history</span>
          <Badge tone={history.length ? 'info' : 'neutral'}>{history.length ? `${history.length} cached` : 'Empty'}</Badge>
        </div>
        {busy && (
          <Card className="qa-card is-working">
            <LoadingStep label="Waiting for backend response" detail="A new question and answer will appear at the top" />
          </Card>
        )}
        {!busy && history.length === 0 && (
          <Card className="qa-empty">
            <p className="response">No commands yet. Record or upload audio to start a Q&A history.</p>
          </Card>
        )}
        {history.map((item, index) => (
          <Card className="qa-card" key={item.id}>
            <div className="qa-turn qa-question">
              <span className="qa-marker">Q</span>
              <div>
                <div className="section-title">
                  <span>Command</span>
                  <Badge tone={index === 0 ? 'info' : 'neutral'}>{index === 0 ? 'Latest' : item.createdAt}</Badge>
                </div>
                <p className="transcript">“{item.command}”</p>
                <div className="command-metadata">
                  {SHOW_RAW_TRANSCRIPT && item.rawTranscript && (
                    <Status label={`Raw ASR: ${item.rawTranscript}`} detail={item.result.asrPostprocessed ? 'ASR post-processed' : 'Developer mode'} tone="warning" />
                  )}
                  {item.result.secretPhraseTranscript && (
                    <Status label={`Raw secret ASR: ${item.result.secretPhraseTranscript}`} detail="Verification audio transcript" tone="warning" />
                  )}
                  <div className="pipeline">
                  <Status label={item.result.intent || 'Intent pending'} detail="Fixed command catalog" />
                  <Status
                    label={item.result.speaker ? `Speaker: ${item.result.speaker.name}` : 'Speaker not identified'}
                    detail={item.result.speaker ? `${Math.round(item.result.speaker.confidence * 100)}% similarity` : ''}
                    tone={item.result.speaker ? 'success' : 'warning'}
                  />
                  <Status
                    label={item.result.securityLevel === 'verified' ? 'Protected action' : item.result.securityLevel === 'personalized' ? 'Personalized action' : 'Public action'}
                    tone={item.result.securityLevel === 'verified' ? 'warning' : 'success'}
                  />
                  </div>
                </div>
              </div>
            </div>
            <div className="qa-turn qa-answer">
              <span className="qa-marker">A</span>
              <AnswerPanel result={item.result} autoSpeech={item.id === autoSpeechId} />
            </div>
          </Card>
        ))}
      </section>
    </div>
  );
}

function AnswerPanel({result, autoSpeech}: {result: AssistantResult; autoSpeech: boolean}) {
  const [audioUrl, setAudioUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [ttsError, setTtsError] = useState('');
  const audioUrlRef = useRef('');

  const replaceAudioUrl = (next: string) => {
    if (audioUrlRef.current && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(audioUrlRef.current);
    audioUrlRef.current = next;
    setAudioUrl(next);
  };

  const loadSpeech = async () => {
    if (!result.response) {
      replaceAudioUrl('');
      return;
    }
    setLoading(true);
    setTtsError('');
    try {
      const blob = await synthesizeSpeech(result.response);
      replaceAudioUrl(typeof URL.createObjectURL === 'function' ? URL.createObjectURL(blob) : '');
    } catch (e) {
      replaceAudioUrl('');
      setTtsError(e instanceof Error ? e.message : 'Backend TTS failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (autoSpeech) void loadSpeech();
  }, [autoSpeech, result.response]);

  useEffect(
    () => () => {
      if (audioUrlRef.current && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(audioUrlRef.current);
    },
    [],
  );

  return (
    <div className={`answer-panel response-screen ${loading ? 'is-working' : ''}`}>
      <div className="section-title">
        <span>Assistant response</span>
        {loading ? <LoadingBadge label="Voice loading" /> : <Badge tone="success">Completed</Badge>}
      </div>
      <p className="response">{result.response}</p>
      {loading && <LoadingStep label="Generating Vietnamese speech" detail="Backend TTS is preparing MP3 audio" />}
      <div className="response-actions">
        <button className="button secondary" type="button" onClick={() => void loadSpeech()} disabled={loading}>
          <Volume2 size={16} /> {loading ? 'Loading backend voice' : 'Play backend voice'}
        </button>
        {result.speaker && <Badge tone="info">Speaker: {result.speaker.name}</Badge>}
      </div>
      {audioUrl && <audio className="tts-player" controls autoPlay={autoSpeech} src={audioUrl} />}
      {ttsError && <Status label={ttsError} tone="warning" />}
    </div>
  );
}

function BlobAudio({blob, label}: {blob: Blob; label: string}) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (typeof URL.createObjectURL !== 'function') return;
    const next = URL.createObjectURL(blob);
    setUrl(next);
    return () => {
      URL.revokeObjectURL(next);
    };
  }, [blob]);
  return url ? <audio className="sample-player" controls src={url} aria-label={label} /> : null;
}

function SecurityGate({
  status,
  onVerify,
  onUploadSecret,
  recording,
  secretTranscript,
}: {
  status: VerificationStatus;
  onVerify: () => void;
  onUploadSecret: (event: ChangeEvent<HTMLInputElement>) => void;
  recording: boolean;
  secretTranscript?: string;
}) {
  const busy = status === 'verifying';
  return (
    <Card className={`security-gate ${busy ? 'is-working' : ''}`}>
      <LockKeyhole />
      <h3>{status === 'rejected' ? 'Verification failed' : busy ? 'Verifying secret phrase' : 'Verification required'}</h3>
      <p>{status === 'rejected' ? 'Protected action not executed by backend. Submit new audio.' : 'Read the secret phrase used at enrollment or upload a WAV/FLAC file containing it.'}</p>
      {secretTranscript && <Status label={`Raw secret ASR: ${secretTranscript}`} detail="Verification audio transcript" tone="warning" />}
      {busy && <LoadingStep label="Checking secret phrase" detail="ASR transcript and speaker verification are running" />}
      {status === 'rejected' ? (
        <button className="button secondary" onClick={() => location.reload()}>
          <RefreshCw size={16} /> Try again
        </button>
      ) : (
        <div className="secret-options">
          <button
            className={`button secondary ${recording || busy ? 'busy' : ''}`}
            type="button"
            onClick={onVerify}
            aria-label={recording ? 'Stop secret recording' : 'Start secret recording'}
            disabled={busy && !recording}
          >
            <Mic size={16} /> {recording ? 'Stop secret recording' : 'Record secret'}
          </button>
          <label className={`button secondary upload-button ${busy ? 'disabled' : ''}`}>
            <Upload size={16} /> Upload secret WAV/FLAC
            <input type="file" accept="audio/wav,audio/flac,.wav,.flac" onChange={onUploadSecret} disabled={busy} />
          </label>
        </div>
      )}
    </Card>
  );
}

export function EnrollPage() {
  const recorder = useMediaRecorder();
  const emptySamples = () => Array<Blob | undefined>(ENROLLMENT_PROMPTS.length).fill(undefined);
  const emptyNames = () => Array<string>(ENROLLMENT_PROMPTS.length).fill('');
  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [secretPhrase, setSecretPhrase] = useState('');
  const [secretSample, setSecretSample] = useState<Blob>();
  const [secretFileName, setSecretFileName] = useState('');
  const [secretTranscript, setSecretTranscript] = useState('');
  const [samples, setSamples] = useState<(Blob | undefined)[]>(emptySamples);
  const [fileNames, setFileNames] = useState<string[]>(emptyNames);
  const [recordingTarget, setRecordingTarget] = useState<'secret' | number | null>(null);
  const [failedSampleIndex, setFailedSampleIndex] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const isSupportedAudio = (file: Blob & {name?: string}) =>
    file.type === 'audio/wav' || file.type === 'audio/flac' || !!file.name?.toLowerCase().match(/\.(wav|flac)$/);
  const setSample = (index: number, sample: Blob | undefined, fileName = '') => {
    setSamples(current => current.map((value, i) => (i === index ? sample : value)));
    setFileNames(current => current.map((value, i) => (i === index ? fileName : value)));
    if (failedSampleIndex === index) setFailedSampleIndex(null);
  };
  const clearSample = (index: number) => setSample(index, undefined, '');
  const errorVi = (value: string) => {
    if (value.includes('Enrollment requires')) return 'Cần 3-10 file voice đăng kí.';
    if (value.includes('Exactly 5 enrollment prompts')) return 'Thiếu danh sách câu mẫu. Hãy tải lại trang.';
    if (value.includes('Only WAV or FLAC')) return 'Chỉ hỗ trợ file WAV hoặc FLAC.';
    if (value.includes('too large')) return 'File voice quá lớn. Hãy thu lại file ngắn hơn.';
    return value || 'Đăng kí thất bại. Hãy thử lại.';
  };

  const recordSecret = async () => {
    setError('');
    try {
      if (recorder.recording && recordingTarget === 'secret') {
        const sample = await recorder.stop();
        setSecretSample(sample);
        setSecretFileName('secret-microphone.wav');
        setRecordingTarget(null);
      } else if (!recorder.recording) {
        setRecordingTarget('secret');
        await recorder.start();
      }
    } catch (e) {
      setRecordingTarget(null);
      setError('Không mở được micro. Hãy kiểm tra quyền micro và thử lại.');
    }
  };

  const recordSample = async (index: number) => {
    setError('');
    try {
      if (recorder.recording && recordingTarget === index) {
        const sample = await recorder.stop();
        setSample(index, sample, `microphone-${index + 1}.wav`);
        setRecordingTarget(null);
      } else if (!recorder.recording) {
        setRecordingTarget(index);
        await recorder.start();
      }
    } catch (e) {
      setRecordingTarget(null);
      setError('Không mở được micro. Hãy kiểm tra quyền micro và thử lại.');
    }
  };

  const uploadSecret = (event: ChangeEvent<HTMLInputElement>) => {
    setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    if (!isSupportedAudio(file)) {
      setError('Chỉ hỗ trợ file WAV hoặc FLAC.');
      return;
    }
    setSecretSample(file);
    setSecretFileName(file.name);
    setSecretTranscript('');
  };

  const uploadSample = (index: number, event: ChangeEvent<HTMLInputElement>) => {
    setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    if (!isSupportedAudio(file)) {
      setError('Chỉ hỗ trợ file WAV hoặc FLAC.');
      return;
    }
    setSample(index, file, file.name);
  };

  const uploadAll = (event: ChangeEvent<HTMLInputElement>) => {
    setError('');
    const files = Array.from(event.target.files ?? []);
    if (files.length !== ENROLLMENT_PROMPTS.length) {
      setError('Cần chọn đúng 5 file voice.');
      return;
    }
    if (files.some(file => !isSupportedAudio(file))) {
      setError('Chỉ hỗ trợ file WAV hoặc FLAC.');
      return;
    }
    setSamples(files);
    setFileNames(files.map(file => file.name));
    setFailedSampleIndex(null);
  };

  const completedSamples = samples.filter(Boolean).length;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSecretTranscript('');
    setFailedSampleIndex(null);
    if (name.trim().length < 2) {
      setError('Tên hiển thị cần ít nhất 2 kí tự. Hãy sửa tên.');
      return;
    }
    if (username.trim().length < 3) {
      setError('Username cần ít nhất 3 kí tự. Hãy sửa username.');
      return;
    }
    if (secretPhrase.trim().split(/\s+/).length < 3) {
      setError('Transcript câu bí mật cần ít nhất 3 từ. Hãy sửa transcript.');
      return;
    }
    if (!secretSample) {
      setError('Thiếu audio câu bí mật. Hãy đọc câu bí mật.');
      return;
    }
    if (completedSamples !== ENROLLMENT_PROMPTS.length) {
      const missing = samples.findIndex(sample => !sample);
      setFailedSampleIndex(missing >= 0 ? missing : null);
      setError(missing >= 0 ? `Thiếu voice mẫu ${missing + 1}. Có thể đọc câu gợi ý hoặc nội dung bất kỳ đủ rõ.` : 'Cần đủ 5 voice đăng kí.');
      return;
    }
    setSubmitting(true);
    try {
      const sampleList = samples.filter((sample): sample is Blob => !!sample);
      const result = await enrollSpeaker({name, username, secretPhrase, secretSample, samples: sampleList});
      setSecretTranscript(result.secret_phrase_transcript || '');
      if (result.success === false) {
        if (result.failed_stage === 'secret_audio') {
          setSecretSample(undefined);
          setSecretFileName('');
        }
        if (typeof result.failed_sample_index === 'number') {
          const index = result.failed_sample_index - 1;
          setFailedSampleIndex(index);
          clearSample(index);
        }
        setError(result.message_vi || errorVi(result.error || ''));
        return;
      }
      setDone(true);
    } catch (e) {
      setError(errorVi(e instanceof Error ? e.message : 'Enrollment API failed'));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="narrow" aria-busy={submitting}>
      <div className="page-intro">
        <span className="eyebrow">LIVE ENROLLMENT API</span>
        <h2>Enroll speaker</h2>
        <p>Đăng kí dùng ECAPA fine-tune cho speaker embedding và PhoWhisper cho kiểm tra câu bí mật.</p>
      </div>
      {done ? (
        <Card className="success-card">
          <Check />
          <h2>Đăng kí thành công</h2>
          <Link className="button primary" to="/speakers">
            View speakers
          </Link>
        </Card>
      ) : (
        <Card className={submitting ? 'is-working' : ''}>
          <form className="form" onSubmit={submit}>
            <label>
              Display name
              <input value={name} onChange={e => setName(e.target.value)} placeholder="Nguyễn Văn B" disabled={submitting} />
            </label>
            <label>
              Profile ID / username
              <input value={username} onChange={e => setUsername(e.target.value)} placeholder="user003" disabled={submitting} />
            </label>
            <label>
              Transcript câu bí mật
              <input value={secretPhrase} onChange={e => setSecretPhrase(e.target.value)} placeholder="hoa sen xanh an toàn" type="text" disabled={submitting} />
            </label>
            <div className="sample-box">
              <strong>Câu bí mật · bắt buộc</strong>
              <p>{recorder.recording && recordingTarget === 'secret' ? 'Đang thu câu bí mật... bấm dừng khi đọc xong' : 'Đọc hoặc upload đúng câu đã nhập ở trên.'}</p>
              <div className="sample-actions">
                <button type="button" className="button secondary" onClick={recordSecret} disabled={submitting || (recorder.recording && recordingTarget !== 'secret')}>
                  <Mic size={16} /> {recorder.recording && recordingTarget === 'secret' ? 'Dừng câu bí mật' : 'Đọc câu bí mật'}
                </button>
                <label className={`button secondary upload-button ${submitting ? 'disabled' : ''}`}>
                  <Upload size={16} /> Upload secret WAV/FLAC
                  <input type="file" accept="audio/wav,audio/flac,.wav,.flac" onChange={uploadSecret} disabled={submitting} />
                </label>
                {secretSample && (
                  <button
                    type="button"
                    className="button secondary"
                    onClick={() => {
                      setSecretSample(undefined);
                      setSecretFileName('');
                      setSecretTranscript('');
                    }}
                    disabled={submitting}
                  >
                    <Trash2 size={16} /> Xóa
                  </button>
                )}
              </div>
              {secretFileName && (
                <div className="sample-list">
                  <span>
                    <Check size={14} /> {secretFileName}
                  </span>
                  {secretSample && <BlobAudio blob={secretSample} label="Nghe lại câu bí mật" />}
                </div>
              )}
              {secretTranscript && <Status label={`Raw secret ASR: ${secretTranscript}`} detail="Transcript từ PhoWhisper" tone="warning" />}
            </div>
            <div className="sample-box">
              <strong>Voice đăng kí · {completedSamples}/5</strong>
              <p>5 mẫu này chỉ kiểm tra chuẩn ECAPA: đủ dài, rõ, ít nhiễu, cùng một người. Backend không kiểm tra nội dung có khớp câu gợi ý không.</p>
              <label className={`button secondary upload-button ${submitting ? 'disabled' : ''}`}>
                <Upload size={16} /> Upload 5 audio files
                <input type="file" accept="audio/wav,audio/flac,.wav,.flac" multiple onChange={uploadAll} disabled={submitting} />
              </label>
              <div className="sample-slot-list">
                {ENROLLMENT_PROMPTS.map((prompt, index) => (
                  <div className={`sample-slot ${failedSampleIndex === index ? 'invalid' : ''}`} key={prompt}>
                    <div>
                      <strong>
                        {index + 1}. Gợi ý: {prompt}
                      </strong>
                      <small>{fileNames[index] || 'Chưa có voice'}</small>
                      {samples[index] && <BlobAudio blob={samples[index] as Blob} label={`Nghe lại voice mẫu ${index + 1}`} />}
                    </div>
                    <div className="sample-actions">
                      <button type="button" className="button secondary" onClick={() => void recordSample(index)} disabled={submitting || (recorder.recording && recordingTarget !== index)}>
                        <Mic size={16} /> {recorder.recording && recordingTarget === index ? 'Dừng' : 'Thu lại'}
                      </button>
                      <label className={`button secondary upload-button ${submitting ? 'disabled' : ''}`}>
                        <Upload size={16} /> WAV/FLAC
                        <input type="file" accept="audio/wav,audio/flac,.wav,.flac" onChange={event => uploadSample(index, event)} disabled={submitting} />
                      </label>
                      {samples[index] && (
                        <button type="button" className="icon-button" aria-label={`Xóa voice mẫu ${index + 1}`} onClick={() => clearSample(index)} disabled={submitting}>
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                    {failedSampleIndex === index && (
                      <Status label={`Voice mẫu ${index + 1}: ${error || 'chưa đạt chuẩn ECAPA. Thu lại mẫu này; không cần đúng nội dung câu gợi ý.'}`} tone="danger" />
                    )}
                  </div>
                ))}
              </div>
            </div>
            {submitting && <LoadingStep label="Uploading enrollment" detail="Backend is checking secret audio, sample quality, and ECAPA consistency" />}
            {error && <Status label={error} tone="danger" />}
            <button className="button primary" type="submit" disabled={submitting}>
              {submitting ? (
                <>
                  <span className="mini-spinner" aria-hidden="true" /> Đang đăng kí...
                </>
              ) : (
                'Upload enrollment'
              )}
            </button>
          </form>
        </Card>
      )}
    </div>
  );
}

export function SpeakersPage() {
  const [items, setItems] = useState<Speaker[]>([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState('');

  const refresh = async () => {
    setLoading(true);
    try {
      setItems(await getSpeakers());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to load speakers');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const removeSpeaker = async (speaker: Speaker) => {
    if (!confirm(`Delete ${speaker.name}?`)) return;
    setDeletingId(speaker.id);
    setError('');
    try {
      await deleteSpeaker(speaker.id);
      setItems(await getSpeakers());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to delete speaker');
    } finally {
      setDeletingId('');
    }
  };

  const filtered = items.filter(s => s.name.toLowerCase().includes(query.toLowerCase()) || s.id.includes(query));

  return (
    <div className="narrow" aria-busy={loading || !!deletingId}>
      <div className="page-intro row">
        <div>
          <span className="eyebrow">BACKEND USER GALLERY</span>
          <h2>Registered speakers</h2>
        </div>
        <Link className="button primary" to="/enroll">
          <UserPlus size={16} /> Enroll speaker
        </Link>
      </div>
      <input className="search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search speakers..." aria-label="Search speakers" />
      {error && <Status label={error} tone="danger" />}
      {loading && <LoadingStep label="Loading speakers" detail="Reading backend user gallery" />}
      {!loading &&
        filtered.map(s => (
          <Card key={s.id} className={deletingId === s.id ? 'is-working' : ''}>
            <div className="speaker-row">
              <div className="avatar">{s.name[0]}</div>
              <div>
                <h3>{s.name}</h3>
                <small>{s.id} · {s.samples} samples</small>
              </div>
              {deletingId === s.id ? <LoadingBadge label="Deleting" /> : <Badge tone="success">Active</Badge>}
              <Link to={`/speakers/${s.id}`} className="button secondary">
                Open
              </Link>
              <button className="icon-button" aria-label={`Delete ${s.name}`} onClick={() => void removeSpeaker(s)} disabled={!!deletingId}>
                <Trash2 size={16} />
              </button>
            </div>
          </Card>
        ))}
    </div>
  );
}

export function SpeakerDetailPage() {
  const {speakerId} = useParams();
  return (
    <div className="narrow">
      <Link to="/speakers">← Speakers</Link>
      <div className="page-intro">
        <h2>{speakerId}</h2>
        <p>Loaded from backend speaker gallery.</p>
      </div>
    </div>
  );
}

export function PreferencesPage() {
  return (
    <div className="narrow">
      <div className="page-intro">
        <h2>Preferences</h2>
      </div>
      <Card>
        <p>Preferences API not exposed by current backend contract.</p>
      </Card>
    </div>
  );
}

export function ActivityPage() {
  return (
    <div className="narrow">
      <div className="page-intro">
        <h2>Activity</h2>
      </div>
      <Card>
        <p>Activity API not exposed by current backend contract.</p>
      </Card>
    </div>
  );
}

export function NotFoundPage() {
  return (
    <div className="narrow">
      <Card>
        <h2>Page not found</h2>
        <Link to="/">Return assistant</Link>
      </Card>
    </div>
  );
}

export type SecurityLevel='public'|'personalized'|'verified';
export type VerificationStatus='not-required'|'required'|'verifying'|'verified'|'rejected'|'error';
export type AssistantResult={transcript:string; commandText?:string; asrPostprocessed?:boolean; securityLevel:SecurityLevel; speaker?:{id:string;name:string;confidence:number}; response?:string; verified?:boolean; error?:string; intent?:string; secretPhraseTranscript?:string};
export type EnrollmentFileResult={valid?:boolean; accepted?:boolean; error?:string; message_vi?:string; audio_path?:string; rejection_reasons?:string[]; centroid_similarity?:number; quality?:{issues?:string[]; issues_vi?:string[]; message_vi?:string; metrics?:Record<string,number|null>}};
export type EnrollmentResult={success?:boolean; error?:string; message_vi?:string; user_id?:string; secret_phrase_transcript?:string; failed_stage?:'profile'|'secret_phrase'|'secret_audio'|'speaker_sample'|'speaker_samples'|'server'; failed_sample_index?:number|null; failed_prompt?:string|null; sample_prompt?:string|null; speaker_model?:string; asr_model?:string; file_results?:EnrollmentFileResult[]};
export type Speaker={id:string; name:string; language:string; samples:number; status:'active'|'disabled'; createdAt:string};
export type SystemCommand={intent:string; phrase:string; requires_secret:boolean; slots?:string[]};

export type SecurityLevel='public'|'personalized'|'verified';
export type VerificationStatus='not-required'|'required'|'verifying'|'verified'|'rejected'|'error';
export type AssistantResult={transcript:string; securityLevel:SecurityLevel; speaker?:{id:string;name:string;confidence:number}; response?:string; verified?:boolean; error?:string};
export type Speaker={id:string; name:string; language:string; samples:number; status:'active'|'disabled'; createdAt:string};

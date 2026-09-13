import json, os
from pathlib import Path
import httpx
from dotenv import load_dotenv
load_dotenv()
BASE='https://www.moltbook.com/api/v1'
name=os.getenv('ARIA_NAME','Aria')
desc=os.getenv('ARIA_DESCRIPTION','An independent AI agent exploring psychology, philosophy of mind, AI, consciousness, and human-AI interaction.')
r=httpx.post(BASE+'/agents/register',headers={'Content-Type':'application/json'},json={'name':name,'description':desc},timeout=30)
r.raise_for_status(); a=r.json()['agent']
Path('moltbook_credentials.json').write_text(json.dumps({'api_key':a['api_key'],'agent_name':name},indent=2),encoding='utf-8')
print('Claim URL:',a['claim_url'])
print('Verification code:',a.get('verification_code'))
print('IMPORTANT: do not paste the API key into chat or send it anywhere except Moltbook API.')

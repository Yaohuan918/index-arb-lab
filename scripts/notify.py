import os, argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--attach", default=None, help="path to file to mention")
    args = ap.parse_args()
    
    webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        # Fallback: just print to stdout so cron/logs show it.
        print(f"[NOTIFY] {args.title}\n{args.test}\nAttachment: {args.attach or '-'}")
        return 
    
    try:
        import requests     # pip install requests
        payload = {"text": f"*{args.title}*\n{args.test}"}
        if args.attach:
            payload["text"] += f"\nAttachment: `{os.path.basename(args.attach)}`"
        r = requests.post(webhook, json=payload, timeout=10)
        r.raise_for_status()
        print("[NOTIFY] OK")
        
    except Exception as e:
        print(f"[NOTIFY] Slack failed: {e}", file=sys.stderr)
        
if __name__ == "__main__":
    main()
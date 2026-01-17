# Deploying Bridge Laws Chatbot to Railway

## Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app - can use GitHub login)
- Your Anthropic API key

## Step 1: Configure Git (if not already done)

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Name it `bridge-laws-chatbot`
3. Keep it **Private** (contains API integration)
4. Don't initialize with README (we already have files)
5. Click "Create repository"

## Step 3: Push to GitHub

Run these commands in the `bridge-laws-chatbot` directory:

```bash
# Commit the files
git add .
git commit -m "Initial commit: Bridge Laws Chatbot"

# Add GitHub remote (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/bridge-laws-chatbot.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Deploy on Railway

1. Go to https://railway.app/dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Find and select `bridge-laws-chatbot`
5. Railway will auto-detect it's a Python project and start building

## Step 5: Add Environment Variable

1. In Railway, click on your deployed service
2. Go to **Variables** tab
3. Click **"+ New Variable"**
4. Add:
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your API key (starts with `sk-ant-`)
5. Railway will auto-redeploy with the new variable

## Step 6: Get Your Public URL

1. Go to **Settings** tab in your Railway service
2. Under **Networking**, click **"Generate Domain"**
3. Railway will give you a URL like `bridge-laws-chatbot-production.up.railway.app`

## Done!

Your chatbot should now be live! Share the URL with others.

---

## Cost Estimate

- **Railway**: Free tier gives you $5/month credit, which is plenty for light usage
- **Claude API**: Pay-per-use. Each question costs ~$0.01-0.03 depending on response length

## To Take It Down

In Railway dashboard, click on your project → Settings → Danger Zone → Delete Service

## Troubleshooting

**Build fails**: Check the Railway logs. Usually missing dependencies in requirements.txt

**API errors**: Make sure ANTHROPIC_API_KEY is set correctly in Railway Variables

**503 errors**: The app may have crashed. Check logs and redeploy.

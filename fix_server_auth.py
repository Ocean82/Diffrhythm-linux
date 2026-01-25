#!/usr/bin/env python3
"""
Fix server authentication to allow testing without Clerk
"""
import sys

with open("src/api/routes.py", "r") as f:
    lines = f.readlines()

# Find the authentication check (around line 579-582)
for i in range(len(lines)):
    if "# Verify authentication" in lines[i] and i + 3 < len(lines):
        # Check if next lines match the pattern
        if "verify_clerk_token" in lines[i+1] and "if not user_id" in lines[i+2]:
            # Get original lines with proper indentation
            verify_line = lines[i+1].rstrip()
            if_line = lines[i+2].rstrip()
            raise_line = lines[i+3].rstrip()
            
            # Ensure proper indentation (12 spaces for else block content)
            if not verify_line.startswith("            "):
                verify_line = "            " + verify_line.lstrip()
            if not if_line.startswith("            "):
                if_line = "            " + if_line.lstrip()
            if not raise_line.startswith("                "):
                raise_line = "                " + raise_line.lstrip()
            
            new_lines = [
                lines[i],  # Keep the comment
                "        # Allow testing without auth if CLERK_SECRET_KEY is not configured\n",
                "        from ..config import settings\n",
                "        # Check if CLERK_SECRET_KEY is None or empty string\n",
                "        clerk_key = getattr(settings, \"CLERK_SECRET_KEY\", None)\n",
                "        if not clerk_key or (isinstance(clerk_key, str) and clerk_key.strip() == \"\"):\n",
                "            # Development/testing mode - allow without authentication\n",
                "            user_id = \"test_user\"\n",
                "            logger.info(\"⚠️  Running in test mode - authentication bypassed\")\n",
                "        else:\n",
                "            # Production mode - require authentication\n",
                verify_line + "\n",
                if_line + "\n",
                raise_line + "\n",
            ]
            # Replace the 4 lines with new code
            lines[i:i+4] = new_lines
            print(f"✅ Modified authentication check at line {i+1}")
            break
    elif i > 600:  # Stop searching after line 600
        print("⚠️  Authentication check not found")
        sys.exit(1)

with open("src/api/routes.py", "w") as f:
    f.writelines(lines)

print("✅ Authentication fix applied successfully")

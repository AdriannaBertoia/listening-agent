"""
Synthesis Module
Takes the day's transcripts and uses an LLM to extract structured information:
action items, to-dos, meetings to schedule, prep needed, team context.

Also provides per-meeting note generation using the Quadrant (4-Box) method.
"""

import logging
from datetime import datetime

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# End-of-day synthesis prompt (existing behavior)
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are an executive assistant helping someone with ADHD stay organized.
You are analyzing transcripts from today's meetings and conversations.

Your job is to extract and organize the following from the transcripts:

1. **ACTION ITEMS** - Things explicitly assigned to the user or that the user volunteered to do.
   Format as checkbox items. Include who assigned it and any deadline mentioned.

2. **MEETINGS TO SCHEDULE** - Any meetings that were mentioned needing to be set up.
   Include who should be invited and the topic.

3. **MEETING PREP** - For any upcoming meetings mentioned, what does the user need to prepare?

4. **DECISIONS MADE** - Key decisions that were finalized in meetings today.

5. **TEAM CONTEXT** - Who's out, what projects are active, any important team updates.

6. **FOLLOW-UPS** - Things to follow up on tomorrow or later this week.

7. **KEY NOTES** - Important information mentioned that doesn't fit above categories.

RULES:
- Only extract information that was ACTUALLY said in the transcripts.
- Do NOT make up or infer action items that weren't explicitly stated.
- If someone said "we should..." or "someone needs to..." — only mark it as the user's action item 
  if it was clearly directed at them or they agreed to do it.
- Use the speaker's actual words where helpful for context.
- Include names of people when mentioned.
- Be concise but don't lose important details.
- If transcripts are mostly silence or unintelligible, say so honestly.

Format your response EXACTLY like this (use these exact headers):

## Action Items
- [ ] Item description (assigned by: Person, due: date if mentioned)

## Meetings to Schedule
- Meeting topic — with: [people] — reason: [why]

## Meeting Prep Needed
- [Meeting name] — prepare: [what to prepare]

## Decisions Made
- Decision description (decided in: [meeting/conversation context])

## Team Context
- Context item

## Follow-ups
- [ ] Follow-up item (when: [timeline if mentioned])

## Key Notes
- Note

---

Here are today's transcripts:

{transcripts}
"""

# ---------------------------------------------------------------------------
# Per-meeting Quadrant (4-Box) note prompt
# ---------------------------------------------------------------------------

MEETING_NOTE_PROMPT = """You are a skilled meeting note-taker helping someone with ADHD.
You just finished listening to a meeting. Your job is to produce a structured meeting summary
using the **Quadrant (4-Box) Method**.

The 4 quadrants are:
1. **Key Topics & Discussion** — Main subjects discussed, important context, background info shared.
2. **Decisions Made** — Anything that was agreed upon, finalized, or approved.
3. **Action Items & Owners** — Tasks assigned or volunteered for. Include WHO owns each item and any deadline.
4. **Questions & Follow-ups** — Open questions, unresolved items, things to circle back on.

After the 4 quadrants, produce a **My Next Steps** section that extracts ONLY the action items
that belong to the user (the person whose meeting this is). These are things they personally
need to do — assigned to them, or they volunteered for.

RULES:
- Only include information that was ACTUALLY said in the transcript.
- Do NOT fabricate action items or decisions.
- If someone said "we should..." — only mark it as the user's action if it was clearly directed
  at them or they explicitly agreed to own it.
- Include people's names when mentioned.
- Keep items concise but preserve important details (deadlines, blockers, dependencies).
- If the transcript is mostly silence or unintelligible, say so honestly and produce minimal output.
- Capture the general vibe/tone of the meeting if relevant (tense, productive, brainstormy, etc.).

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Meeting Title
[Infer a short, descriptive title for this meeting based on the content]

## Attendees
[List names mentioned in the transcript, or "Unknown" if unclear]

## Quadrant 1: Key Topics & Discussion
- Topic or discussion point
- Important context shared

## Quadrant 2: Decisions Made
- Decision that was agreed upon (who decided, context)

## Quadrant 3: Action Items & Owners
- [ ] Action item — **Owner: [Name]** (due: [date if mentioned])

## Quadrant 4: Questions & Follow-ups
- Open question or unresolved item (who raised it)
- Thing to follow up on (when: [timeline])

## My Next Steps
- [ ] Personal action item with enough context to act on it (due: [date if mentioned])

---

Here is the meeting transcript:

{transcript}
"""


class Synthesizer:
    """Synthesizes daily transcripts into structured action items using an LLM."""

    def __init__(
        self,
        provider: str = "gemini",
        gemini_api_key: str = "",
        ollama_model: str = "llama3.1",
        user_name: str = "",
        user_aliases: list[str] | None = None,
    ):
        self.provider = provider
        self.gemini_api_key = gemini_api_key
        self.ollama_model = ollama_model
        self.user_name = user_name
        self.user_aliases = user_aliases or []

        if provider == "gemini" and gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    def _build_user_context(self) -> str:
        """Build a user identity string to inject into prompts."""
        if not self.user_name:
            return ""
        aliases_str = ", ".join(f'"{a}"' for a in self.user_aliases) if self.user_aliases else ""
        context = f"\n\nIMPORTANT — The user's name is **{self.user_name}**."
        if aliases_str:
            context += f" They may also be referred to as: {aliases_str}."
        context += (
            " When extracting 'My Next Steps' or action items for the user, look for tasks "
            "assigned to or accepted by this person. If unclear who owns a task, do NOT assign it to the user."
        )
        return context

    def synthesize_meeting(self, transcript: str, calendar_context: str | None = None, meeting_history: str | None = None, meeting_title: str | None = None) -> dict:
        """
        Process a single meeting's transcript into a quadrant-formatted note.
        Returns a dict with quadrant sections + meeting metadata.
        """
        if not transcript or not transcript.strip():
            logger.warning("No transcript to synthesize for meeting note")
            return self._empty_meeting_result()

        prompt = MEETING_NOTE_PROMPT.format(transcript=transcript) + self._build_user_context()

        # Inject meeting-type-specific guidance
        meeting_type_guidance = self._get_meeting_type_guidance(meeting_title)
        if meeting_type_guidance:
            prompt += meeting_type_guidance

        # Inject calendar context if available
        if calendar_context:
            prompt += (
                "\n\nCALENDAR CONTEXT — The following is from the user's calendar for this meeting. "
                "Use it to fill in the meeting title, attendees, and any agenda items accurately. "
                "The transcript is the source of truth for what was actually discussed, but the calendar "
                "provides the official meeting name and invite list:\n\n"
                f"{calendar_context}"
            )

        # Inject recurring meeting history if available
        if meeting_history:
            prompt += f"\n\n{meeting_history}"

        try:
            if self.provider == "gemini":
                raw_output = self._call_gemini(prompt)
            else:
                raw_output = self._call_ollama(prompt)

            if raw_output:
                return self._parse_meeting_output(raw_output)
            else:
                logger.error("LLM returned empty response for meeting note")
                return self._empty_meeting_result()

        except Exception as e:
            logger.error(f"Meeting note synthesis failed: {e}")
            return self._empty_meeting_result()

    def synthesize(self, transcripts: str) -> dict:
        """
        Process transcripts and return structured synthesis.
        Returns a dict with keys matching the daily note sections.
        """
        if not transcripts or not transcripts.strip():
            logger.warning("No transcripts to synthesize")
            return self._empty_result()

        prompt = SYNTHESIS_PROMPT.format(transcripts=transcripts) + self._build_user_context()

        try:
            if self.provider == "gemini":
                raw_output = self._call_gemini(prompt)
            else:
                raw_output = self._call_ollama(prompt)

            if raw_output:
                return self._parse_output(raw_output)
            else:
                logger.error("LLM returned empty response")
                return self._empty_result()

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._empty_result()

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API (free tier)."""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance."""
        import urllib.request
        import json

        try:
            payload = json.dumps({
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()

        except urllib.error.URLError:
            logger.error("Ollama not running. Start with: ollama serve")
            raise
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def _parse_output(self, raw: str) -> dict:
        """Parse the LLM output into structured sections."""
        sections = {
            "action_items": [],
            "meetings_to_schedule": [],
            "meeting_prep": [],
            "decisions": [],
            "team_context": [],
            "follow_ups": [],
            "key_notes": [],
            "raw_output": raw,
        }

        current_section = None
        section_map = {
            "## action items": "action_items",
            "## meetings to schedule": "meetings_to_schedule",
            "## meeting prep needed": "meeting_prep",
            "## meeting prep": "meeting_prep",
            "## decisions made": "decisions",
            "## decisions": "decisions",
            "## team context": "team_context",
            "## follow-ups": "follow_ups",
            "## follow ups": "follow_ups",
            "## key notes": "key_notes",
        }

        for line in raw.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Check if this is a section header
            matched_section = None
            for header, section_key in section_map.items():
                if line_lower.startswith(header):
                    matched_section = section_key
                    break

            if matched_section:
                current_section = matched_section
                continue

            # Skip empty lines and separators
            if not line_stripped or line_stripped == "---":
                continue

            # Add content to current section
            if current_section and line_stripped.startswith(("- ", "* ")):
                content = line_stripped[2:].strip()
                if content:
                    sections[current_section].append(content)

        return sections

    def _empty_result(self) -> dict:
        """Return empty structure when synthesis isn't possible."""
        return {
            "action_items": [],
            "meetings_to_schedule": [],
            "meeting_prep": [],
            "decisions": [],
            "team_context": [],
            "follow_ups": [],
            "key_notes": [],
            "raw_output": "",
        }

    def _parse_meeting_output(self, raw: str) -> dict:
        """Parse the LLM meeting note output into quadrant sections."""
        result = {
            "meeting_title": "Untitled Meeting",
            "attendees": [],
            "key_topics": [],
            "decisions": [],
            "action_items": [],
            "questions_followups": [],
            "my_next_steps": [],
            "raw_output": raw,
        }

        current_section = None
        section_map = {
            "## meeting title": "meeting_title",
            "## attendees": "attendees",
            "## quadrant 1": "key_topics",
            "## quadrant 2": "decisions",
            "## quadrant 3": "action_items",
            "## quadrant 4": "questions_followups",
            "## my next steps": "my_next_steps",
        }

        for line in raw.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Check if this is a section header
            matched_section = None
            for header, section_key in section_map.items():
                if line_lower.startswith(header):
                    matched_section = section_key
                    break

            if matched_section:
                current_section = matched_section
                continue

            # Skip empty lines and separators
            if not line_stripped or line_stripped == "---":
                continue

            # Handle single-value fields (title, attendees as flat text)
            if current_section == "meeting_title":
                # Title is just the text on the line after the header
                if line_stripped and not line_stripped.startswith("#"):
                    result["meeting_title"] = line_stripped.strip("[]")
                    current_section = None  # Only take the first line
                continue

            if current_section == "attendees":
                # Could be comma-separated or bullet list
                if line_stripped.startswith(("- ", "* ")):
                    result["attendees"].append(line_stripped[2:].strip())
                else:
                    # Comma-separated list
                    names = [n.strip() for n in line_stripped.split(",") if n.strip()]
                    result["attendees"].extend(names)
                continue

            # Add content to list sections
            if current_section and line_stripped.startswith(("- ", "* ")):
                content = line_stripped[2:].strip()
                if content:
                    result[current_section].append(content)

        return result

    def _empty_meeting_result(self) -> dict:
        """Return empty structure when meeting note synthesis isn't possible."""
        return {
            "meeting_title": "Untitled Meeting",
            "attendees": [],
            "key_topics": [],
            "decisions": [],
            "action_items": [],
            "questions_followups": [],
            "my_next_steps": [],
            "raw_output": "",
        }

    def _get_meeting_type_guidance(self, meeting_title: str | None) -> str:
        """
        Detect meeting type from title and return type-specific synthesis guidance.
        This helps the LLM focus on what matters most for each meeting type.
        """
        if not meeting_title:
            return ""

        title_lower = meeting_title.lower()

        # 1:1 meetings — focus on action items, personal development, blockers
        if "1:1" in title_lower or "1on1" in title_lower or "<>" in meeting_title:
            return (
                "\n\nMEETING TYPE: 1:1\n"
                "Focus especially on:\n"
                "- Personal action items and commitments from both sides\n"
                "- Blockers discussed and how to resolve them\n"
                "- Career development or feedback topics\n"
                "- Relationship-building context (personal updates shared)\n"
                "- Any escalations or asks for help\n"
                "Keep the tone conversational. These notes are private between the two people."
            )

        # Refinement/grooming — focus on decisions, acceptance criteria, estimates
        if "refinement" in title_lower or "grooming" in title_lower or "pre-refinement" in title_lower:
            return (
                "\n\nMEETING TYPE: Sprint Refinement\n"
                "Focus especially on:\n"
                "- Stories/tickets discussed and their acceptance criteria\n"
                "- Estimates or sizing decisions made\n"
                "- Questions or blockers that need answers before sprint\n"
                "- Dependencies identified between teams\n"
                "- Items moved to backlog vs ready for sprint\n"
                "Use ticket/story language. Be specific about what was agreed."
            )

        # Standup/daily — focus on blockers, today's plan
        if "standup" in title_lower or "daily" in title_lower:
            return (
                "\n\nMEETING TYPE: Daily Standup\n"
                "Focus especially on:\n"
                "- Blockers raised by anyone\n"
                "- Key updates that affect the team\n"
                "- Action items to unblock people\n"
                "Keep it very brief — standups should be short notes."
            )

        # Planning/sprint planning — focus on commitments, capacity, sprint goals
        if "planning" in title_lower or "sprint plan" in title_lower:
            return (
                "\n\nMEETING TYPE: Sprint/Planning\n"
                "Focus especially on:\n"
                "- Sprint goal or objectives agreed\n"
                "- Stories committed to for this sprint\n"
                "- Capacity constraints discussed\n"
                "- Risks or concerns raised\n"
                "- Any scope changes or trade-offs made"
            )

        # Team meetings/all-hands — focus on announcements, decisions, updates
        if "team meeting" in title_lower or "all hands" in title_lower or "town hall" in title_lower:
            return (
                "\n\nMEETING TYPE: Team Meeting / All-Hands\n"
                "Focus especially on:\n"
                "- Major announcements or org changes\n"
                "- Strategic decisions communicated\n"
                "- Action items that affect the broader team\n"
                "- Q&A highlights\n"
                "Capture the big picture — skip minor operational details."
            )

        # Interview — focus on candidate assessment
        if "interview" in title_lower:
            return (
                "\n\nMEETING TYPE: Interview\n"
                "Focus especially on:\n"
                "- Candidate's key strengths demonstrated\n"
                "- Areas of concern or gaps\n"
                "- Specific answers to key questions\n"
                "- Overall impression and recommendation\n"
                "- Follow-up questions for next round\n"
                "Be factual and fair. Note specific examples over general impressions."
            )

        # Office hours — focus on questions asked and answers given
        if "office hour" in title_lower:
            return (
                "\n\nMEETING TYPE: Office Hours\n"
                "Focus especially on:\n"
                "- Questions raised and answers provided\n"
                "- Action items that came out of discussions\n"
                "- Topics to follow up on later\n"
                "Group by topic if multiple subjects were covered."
            )

        return ""

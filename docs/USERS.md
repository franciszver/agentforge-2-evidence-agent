# Users

Who the Clinical Co-Pilot is built for, and the five questions it exists to answer.

## Persona

A primary-care physician running a ~20-patient day, with roughly 90 seconds between
one exam room and the next. In the room, they work from a desktop workstation —
that's where OpenEMR itself lives. Everywhere else — the hallway between rooms, the
stairwell, a home-call line at 11pm — the only device in reach is the phone in their
pocket. That gap is not incidental: it's why mobile-first is a hard requirement for
this project, not a nicety bolted on later.

Their motivation in the 90-second window is narrow and concrete: get oriented on the
patient they're about to see, or resolve one specific question about a patient they
just saw, without derailing into a multi-click chart review. Success feels like
walking into the room already knowing what changed, what the patient is on, and
whether starting a new medication is safe — not toggling between five OpenEMR
screens while the patient waits. The constraint is never "give me more data"; it's
"give me the one fact I need, verified, before the door opens."

## Use Cases

| UC | Use case | In the physician's words | Why an agent, not a dashboard |
|---|---|---|---|
| UC1 | Pre-visit brief | "What changed since I last saw her?" | The answer spans encounters, labs, meds, and notes — a synthesis task no single OpenEMR screen produces; it requires combining sources, not just sorting one. |
| UC2 | Medication safety | "What is she taking, and does anything conflict with starting ibuprofen?" | Requires cross-referencing active medications against allergies and interaction data, then citing each source — a check, not a lookup. |
| UC3 | Lab trend recall | "What are her last three A1c values, and when?" | Trivial to state, tedious to click through — stock OpenEMR takes 4+ navigations to assemble the same answer a sentence can give. |
| UC4 | Conversational follow-up drill-down | "Which visit was that from?" → "Show me the note." | Inherently conversational: the second question only makes sense in light of the first answer. A dashboard has no memory of what you just asked. |
| UC5 | Hallway recall (mobile) | Same questions as UC1–UC4, asked from a phone over Tailscale before walking into the room | The 90-second window most often happens away from the workstation — if the answer isn't reachable from a phone, it isn't reachable in time. |

## Design Principle

Every tool the Co-Pilot builds maps to one of UC1–UC5 above. No capability ships
without a use case behind it.

# Conceptual Approaches to NBA Minutes Modeling

**Objective**: Propose 3 fundamentally different conceptual approaches to modeling NBA minutes that inherently resist "smearing" (under-allocating stars, over-allocating fringe) and do not rely on direct minutes prediction (regression).

---

## Approach 1: The "Trust-Based" Survival Model (Time-to-Event)

### The Concept
Instead of asking "How many minutes will Player X play?", this approach asks **"Given Player X is on the court, what is the probability they get subbed out in the next minute?"**

This frames minutes as a **Survival Analysis** problem. Every stint on the court is a life; a substitution is death. The total minutes are simply the sum of the expected durations of all "lives" (stints).

### The Latent Variable: "The Leash" (Substitution Hazard Rate)
The core latent variable is the **Hazard Rate** $h(t)$, or "The Leash."
*   **Stars (LeBron, Luka)** have a flat, near-zero hazard rate for long stretches. The coach does not sub them unless they signal for rest or the quarter ends. Their "leash" is infinite.
*   **Role Players** have a moderate hazard rate that increases comfortably with fatigue.
*   **Fringe Players** have a "spike" hazard rate. One mistake, one missed rotation, or simply the starter catching their breath, and they are pulled. Their "leash" is extremely short.

### Why it Succeeds Where Regression Fails
Regression models "regress to the mean," pulling high predictions down and low predictions up—the definition of smearing.
Survival models naturally produce **heavy-tailed distributions**. If the hazard rate is close to zero (as it is for stars), the expected duration explodes linearly. It doesn't "dampen" the minutes; it allows them to run until a constraint (end of game/fatigue) hits.
It inherently captures the asymmetry of trust: "It is hard to get Lebron OFF the floor; it is hard to keep a Rookie ON the floor."

### Biggest Failure Mode
**Context-Dependence (Foul Trouble / Blowouts).**
Survival models assume the "event" (substitution) follows a probabilistic law based on the player's intrinsic properties. However, many substitutions are forced by external state violations (2 fouls in Q1, game is a blowout).
*   *Blind Spot*: If the model learns a "low hazard" for a star, it might stubbornly leave them on the floor during a 30-point blowout or when they have 5 fouls, because it hasn't seen enough "failure events" for that specific context to adjust the hazard rate.

---

## Approach 2: The "Possession Economy" (Resource Constraint Optimization)

### The Concept
Minutes are not the resource; **Possessions** are the resource. The team has a finite budget of ~100 offensive possessions per game.
Instead of allocating time, the coach allocates **Production Opportunities**. Minutes are just a derived byproduct of the mathematical necessity to field a lineup that can absorb these 100 possessions effectively.

### The Latent Variable: "Usage Capacity" (or Market Share)
The core latent variable is **Usage Capacity** (the ability to consume possessions efficiently).
*   **The Constraint**: You cannot play 5 low-usage players (sum of usage < 100%) because you will turn the ball over or take violation penalties. You cannot play 5 ball-dominant stars (sum of usage > 100%) because there is only one ball.
*   **The Mechanism**: The model serves as an Optimizer. It fills the bucket of 100 possessions with the most efficient high-volume options first (Stars). As the Stars "fill up" their fatigue limits, the model must "buy" possessions from the Bench.
*   **Result**: Stars play maximum minutes because they are the only ones who can "afford" to use the possessions. Fringe players only play when the "Possession Market" is underserved (garbage time or resting stars).

### Why it Succeeds Where Regression Fails
It enforces **Concentration** by definition. A team *must* lean on its primary creators to function. Smearing happens in regression because the model treats Player 10 and Player 1 as interchangeable units of time. In the Economy model, Player 1 (35% Usage) and Player 10 (12% Usage) are different currencies. You cannot swap them without breaking the "Possession Sum" constraint. This structural constraint forces the model to keep the Star on the floor to maintain equilibrium.

### Biggest Failure Mode
**The "Cardio Specialist" (PJ Tucker Problem).**
Some players are elite specifically because they play high minutes while consuming *zero* resources (corner spacers, defensive specialists).
*   *Blind Spot*: The Economy model will view a 0% Usage defensive ace as "worthless" or "unnecessary" to the possession math and might project them for 0 minutes. It struggles to value "non-consumptive" contribution.

---

## Approach 3: The "Slot-Filling" Generative Syntax (Discrete Structure)

### The Concept
Basketball rotations are not continuous fluids; they are **Discrete Syntactical Structures**. Coaches use templates: "The 9-Man Rotation," "The Hockey Sub," "The Staggered Stars."
This approach treats rotation modeling as **NLP (Natural Language Processing)** or **Tagging**. The game is a sentence; the minutes are the grammar. We don't predict "24.5 minutes"; we predict "Role: Sixth Man".

### The Latent Variable: "The Role ID" (Archetype Slot)
The core latent variable is the **Role Slot** (e.g., Slot A1: Primary Ballhandler, Slot B1: Backup Center).
*   **Step 1:** Classify the Coach’s "Game Plan Template" (e.g., "Thibodeau Strict 9-Man").
*   **Step 2:** This template has rigid, pre-defined slots with fixed minute distributions (Star: 38m, Starter: 30m, 6th Manness: 24m, Fringe: 0m).
*   **Step 3:** Perform a "Bipartite Matching" problem to assign the available roster of players to these fixed Slots.
*   **Result:** A player is assigned "Slot A1" and inherits 38 minutes exactly. There is no smearing because the *slot* is sharp, even if the *assignment* is probabilistic.

### Why it Succeeds Where Regression Fails
It explicitly forbids the "middle ground." In a regression, if you are unsure if a player is a Starter (30m) or Bench (15m), it averages them to 22.5m (which is wrong in both worlds).
In Syntax modeling, the model must choose: Is he Slot A or Slot B? If it picks Slot A, he gets 30m. If Slot B, 15m. The *output distribution* is multi-modal and sharp, matching the reality of coaching decisions which are binary/categorical in nature.

### Biggest Failure Mode
**Rigidity / Lack of Expressiveness.**
Coaches are creatures of habit until they aren't.
*   *Blind Spot*: When a coach invents a new rotation (e.g., "Triple Big" or "Play the hot hand") that doesn't fit a pre-learned "Template," the model has no slot to represent it. It will force a round peg into a square hole, potentially mis-assigning a role completely because the "correct" role doesn't exist in its vocabulary.

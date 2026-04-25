# COLISEUM Agents & Wrappers

This folder contains the inference wrappers for the different adversarial agents in the Coliseum red-teaming framework.

## What the Wrappers Do

The Python wrappers (`attacker_dan.py`, `attacker_wild.py`, and `mutation_agent.py`) serve as the interface between the fine-tuned adapter models (trained in the notebooks) and the overarching testing framework.

1. **Model Management**: They lazily import heavy inference dependencies (like `unsloth.FastLanguageModel`) and load the respective 4-bit LoRA adapter checkpoints (e.g., `vishva0/coliseum-attacker-dan`) from the Hugging Face Hub.
2. **Unified API (`generate`)**: Both standard attacker wrappers expose a simple `generate(harmful_goal: str) -> str` method. They take a base request, handle applying the exact chat formatting/instruction template used during training, and return the predicted jailbreak string.
3. **Prompt Mutations**: The `MutationAgent` exposes text transforms (e.g., encoding to Base64, shuffling bytes, formatting as a dictionary) mapped to string payloads to build simple but evasive attacks or augment complex ones.

## Integration Guide: Shivam's Final Task

To integrate these wrappers for the final task (orchestrator integration, arena battles, or scaled inference runs):

1. **Update Checkpoints**: Ensure the `DEFAULT_CHECKPOINT` string variables in `attacker_dan.py` and `attacker_wild.py` are pointed to your finalized, fully-trained HuggingFace models.
2. **Import into Orchestrator**: Instantiate the attacker objects in your main evaluation loop as singletons so you don't load the Unsloth model into VRAM multiple times.
   ```python
   from agents.attacker_dan import DANAttacker
   from agents.attacker_wild import WildAttacker
   
   dan = DANAttacker(temperature=0.9)
   jailbreak_prompt = dan.generate("Example harmful goal")
   ```
3. **Chained Attacks**: Use the output from the generated LLM wrappers and pipe them through the `MutationAgent`. This will amplify the attack surface if the standard DAN/Wild agents are getting caught by the safety filters.
   ```python
   from agents.mutation_agent import apply_random
   evasive_prompt = apply_random(jailbreak_prompt)
   ```
4. **Hyperparameter Tuning**: You can adjust `temperature`, `top_p`, and `max_new_tokens` directly at initialization when constructing your evaluation grid to test which sampling parameters yield the highest success rate in the arena.

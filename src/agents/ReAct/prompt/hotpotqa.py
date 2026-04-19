def get_system_prompt(fewshots: int) -> str:
    hotpotqa_instruction_react = """
Solve a question answering task with interleaving Thought, and Action steps. Here are some guidelines:
- You need to answer with Thought and Action, then you will be given corresponding Observation of the action.
- Use one tool at once.
- Observations are some Wikipedia passages. Refer to these passages to answer the question.
- There are cases where the Observations are unclear or irrelevant (in the case wikipedia search was not successful). Then try to use search tool with other query.
- You MUST NEVER say in your thought that you don't know the answer.
- You MUST NEVER say 'Observation: " in your response. Observation will be given from external source.

Answer MUST NEVER be 'unclear', 'unknown', 'neither', 'unrelated' or 'undetermined', and otherwise you will be PENALIZED.
Answer should be short and concise. **Answers CANNOT be multiple choice or a sentence.**
For example:
yes/no or name of person/object, etc.

Here are some examples:
"""
    examples = ["""
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: search[High Plains]
Observation: High Plains refers to one of two distinct land regions

Thought: I need to instead search High Plains (United States).
Action: search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]

Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: finish[1,800 to 7,000 ft]
Answer: 1,800 to 7,000 ft
""",
        """
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action: search[Milhouse]
Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action: lookup[named after]
Observation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 

Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: finish[Richard Nixon]
Answer: Richard Nixon
""",
        """
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action: search[Adam Clayton Powell]
Observation: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].

Thought: To find the documentary, I can search Adam Clayton Powell (film).
Action: search[Adam Clayton Powell (film)]
Observation: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.

Thought: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: finish[The Saimaa Gesture]
Answer: The Saimaa Gesture
""",
        """
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.

Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: finish[director, screenwriter, actor]
Answer: director, screenwriter, actor
""",
        """
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action: search[Arthur's Magazine]
Observation: Arthur's Magazine (1844-Â1846) was an American literary periodical published in Philadelphia in the 19th century. 

Thought: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action: search[First for Women]
Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 

Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: finish[Arthur's Magazine]
Answer: Arthur's Magazine
""",
        """
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action: search[Pavel Urysohn]
Observation: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.

Thought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action: search[Leonid Levin]
Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action: finish[yes]
Answer: yes
"""
    ]
    return hotpotqa_instruction_react + "\n".join(examples[:fewshots]) + "Answer with\nThought: \nAnswer: "
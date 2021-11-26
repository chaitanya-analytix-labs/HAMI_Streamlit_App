# ---------- Use Case Exception Config ----------
# Specify a number form 1 ~ 10, higher is more strict
StrictLevel = 1
# Confirmation target sentence
ConfirmationTargetSentence = 'may I have a phone confirmation to proceed with the application?'
# Confirmation key words in agent's sentences
ConfirmationAgentKeyWords = ['confirm', 'confirmation', 'conversation']
# Confirmation response key words in customer's sentences
ConfirmationCustomerKeyWords = ['yes', 'yeah', 'sure', 'ok', 'can', 'correct', 'accept', 'thank']
# Verification target sentence
VerificationTargetSentence = 'please allow me to do a quick verification'
# Verification key words in agent's sentences
VerificationAgentKeyWords = ['verify', 'verification', 'nric', 'ic', 'number']
# Verification response key words in customer's sentences
VerificationCustomerKeyWords = ['nric', 'ic', 'number', 'birth']



# ---------- Text/phrase identification Config ----------
# Specify a number form 1 ~ 10, higher is more strict
StrictLevel = 1
# target sentence
TargetSentence = 'undermining national unity'
# key words in sentences
KeyWords = ['public safety', 'public discipline', 'racial hostility','Sedition',
    'seditious','religious values', 'religious sentiment', 'malicious', 'dishonest intention', 'panic',
    'islam', 'insults', 'race', 'violence']


# ---------- Result -----------
text=["""Offence and punishment for committing cyber terrorism.¾(1) If any person¾ (a) 
creates obstruction to make legal access, or makes or causes to make illegal access to any computer or 
computer network or internet network with an intention to jeopardize the integrity, security and 
sovereignty of the State and to create a sense of fear or panic in the public or a section of the public; or 
(b) creates pollution or inserts malware in any digital device which may cause or likely to cause death or serious injury to a person; or 
(c) affects or damages the supply and service of daily commodity of public or creates adverse effect on any critical information infrastructure; 
or (d) intentionally or knowingly gains access to, or makes interference with, any computer, computer network, internet network, 
any protected data-information or computer database, or gains access to any such protected data information 
or computer database which may be used against friendly relations with another foreign country or public order, 
or may be used for the benefit of any foreign country or any individual or any group, 
then such person shall be deemed to have committed an offence of cyber terrorism."""]

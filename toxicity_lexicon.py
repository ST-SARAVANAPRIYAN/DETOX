"""
Toxicity Lexicon - Curated word lists for feature engineering
==============================================================
Defines toxic word categories for enhanced detection accuracy
"""

# Profanity & Offensive Language
PROFANITY = {
    'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
    'shit', 'shitty', 'bullshit', 'shits',
    'damn', 'damned', 'dammit',
    'ass', 'asshole', 'asses', 'assholes',
    'bitch', 'bitches', 'bitching',
    'bastard', 'bastards',
    'crap', 'crappy',
    'piss', 'pissed', 'pissing',
    'dick', 'dickhead', 'dicks',
    'cock', 'cunt',
    'motherfucker', 'motherfucking',
    'jackass', 'dumbass',
}

# Threats & Violence
THREATS = {
    'die', 'kill', 'killing', 'killed', 'death', 'dead',
    'murder', 'murderer',
    'shoot', 'shooting', 'shot',
    'stab', 'stabbing', 'stabbed',
    'attack', 'attacking', 'attacked',
    'destroy', 'destroying', 'destroyed',
    'hurt', 'hurting', 'harm', 'harming',
    'beat', 'beating',
    'rape', 'raped', 'raping',
    'torture', 'tortured',
}

# Hate Speech & Slurs (carefully curated - avoiding false positives)
HATE_SPEECH = {
    'nigger', 'nigga', 'negro',
    'fag', 'faggot', 'faggots',
    'retard', 'retarded', 'retards',
    'kike', 'kikes',
    'nazi', 'nazis',
    'terrorist', 'terrorists',
    'scum', 'trash', 'garbage',
    'vermin', 'subhuman',
}

# Insults & Demeaning Terms
INSULTS = {
    'stupid', 'idiot', 'idiots', 'idiotic',
    'moron', 'moronic', 'morons',
    'dumb', 'dumber', 'dumbest',
    'loser', 'losers',
    'pathetic', 'worthless',
    'ugly', 'disgusting',
    'ignorant', 'ignorance',
    'retard', 'retarded',
    'fool', 'foolish', 'fools',
    'scum', 'trash',
    'waste', 'useless',
    'failure', 'failures',
    'joke', 'clown',
}

# Aggressive & Hostile Terms
AGGRESSIVE = {
    'hate', 'hating', 'hated', 'hates',
    'shut up', 'shutup',
    'go to hell', 'hell',
    'fuck off', 'piss off',
    'get lost',
    'suck', 'sucks', 'sucking',
    'lame', 'terrible', 'awful',
    'gtfo', 'stfu',
}

# Severity Weights
SEVERITY_WEIGHTS = {
    'EXTREME': 3.0,  # Severe threats, slurs
    'HIGH': 2.0,     # Strong profanity, hate speech
    'MEDIUM': 1.5,   # Regular profanity, insults
    'LOW': 1.0,      # Mild aggressive terms
}

# Categorize words by severity
EXTREME_TOXIC = THREATS | HATE_SPEECH
HIGH_TOXIC = PROFANITY
MEDIUM_TOXIC = INSULTS
LOW_TOXIC = AGGRESSIVE

# All toxic words combined
ALL_TOXIC_WORDS = EXTREME_TOXIC | HIGH_TOXIC | MEDIUM_TOXIC | LOW_TOXIC


def get_toxic_word_count(text: str) -> dict:
    """
    Count toxic words by category
    
    Args:
        text: Lowercase cleaned text
        
    Returns:
        Dictionary with counts and severity score
    """
    words = set(text.lower().split())
    
    extreme_count = len(words & EXTREME_TOXIC)
    high_count = len(words & HIGH_TOXIC)
    medium_count = len(words & MEDIUM_TOXIC)
    low_count = len(words & LOW_TOXIC)
    
    total_toxic = extreme_count + high_count + medium_count + low_count
    total_words = len(text.split())
    
    # Calculate weighted severity score
    severity_score = (
        extreme_count * SEVERITY_WEIGHTS['EXTREME'] +
        high_count * SEVERITY_WEIGHTS['HIGH'] +
        medium_count * SEVERITY_WEIGHTS['MEDIUM'] +
        low_count * SEVERITY_WEIGHTS['LOW']
    )
    
    return {
        'extreme_toxic_count': extreme_count,
        'high_toxic_count': high_count,
        'medium_toxic_count': medium_count,
        'low_toxic_count': low_count,
        'total_toxic_count': total_toxic,
        'total_words': total_words if total_words > 0 else 1,
        'toxic_word_ratio': total_toxic / total_words if total_words > 0 else 0.0,
        'severity_score': severity_score
    }


def is_highly_toxic(text: str, threshold: float = 2.0) -> bool:
    """
    Quick check if text is highly toxic based on lexicon
    
    Args:
        text: Input text
        threshold: Severity score threshold
        
    Returns:
        True if severity score >= threshold
    """
    stats = get_toxic_word_count(text.lower())
    return stats['severity_score'] >= threshold


# Export main components
__all__ = [
    'ALL_TOXIC_WORDS',
    'EXTREME_TOXIC',
    'HIGH_TOXIC',
    'MEDIUM_TOXIC',
    'LOW_TOXIC',
    'get_toxic_word_count',
    'is_highly_toxic',
]

def get_task(name, file=None):
    if name == 'trivia_creative_writing':
        from .trivia_creative_writing import TriviaCreativeWritingTask
        return TriviaCreativeWritingTask(file)
    elif name == 'logic_grid_puzzle':
        from .logic_grid_puzzle import LogicGridPuzzleTask
        return LogicGridPuzzleTask(file)
    elif name == 'codenames_collaborative':
        from .codenames_collaborative import CodenamesCollaborativeTask
        return CodenamesCollaborativeTask(file)
    elif name == 'sonnets':
        from .sonnets import sonnetsTask
        return sonnetsTask(file)
    elif name == 'word_sorting':
        from .word_sorting import word_sortingTask
        return word_sortingTask(file)
    elif name == 'multistep_arithmetic_two':
        from .multistep_arithmetic_two import multistep_arithmetic_twoTask
        return multistep_arithmetic_twoTask(file)
    elif name == 'qasc':
        from .qasc import qascTask
        return qascTask(file)
    else:
        raise NotImplementedError
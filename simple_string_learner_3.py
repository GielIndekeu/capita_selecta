import typing
from abc import ABC, abstractmethod

from orderedset import OrderedSet
from pylo.language.commons import Functor

from loreleai.language.lp import c_pred, Clause, Procedure, Atom, c_var, c_const, Body, List, Pair, list_func
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
from simple_learning_system import TemplateLearner


class SimpleBreadthFirstLearner(TemplateLearner):

    def __init__(self, solver_instance: Prolog, max_body_literals=4):
        super().__init__(solver_instance)
        self._max_body_literals = max_body_literals

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        if isinstance(candidates, Clause):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)

    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        """
        This function differs from the origional in that it does not check which examples are covered, but checks
        instead rather or not the clause has a solution
        """
        print("evaluating: " + str(clause))
        if len(clause.get_body().get_literals()) == 0:
            return 0
        has_solution = self._solver.has_solution(*clause.get_body().get_literals())
        if not has_solution:
            return 0
        self._solver.assertz(clause=clause)
        pos, neg = examples.get_examples()
        for example in neg:
            if self._solver.query(example):
                self._solver.retract(clause=clause)
                return 0
        covered = 0
        for example in pos:
            if self._solver.query(example):
                covered += 1
        print("success:  " + str(covered) + " pos examples covered")
        return covered
        #covered = self._execute_program(clause)
        """
        pos, neg = examples.get_examples()
        covered_pos = pos.intersection(covered)
        covered_neg = neg.intersection(covered)
        if (len(covered_pos) != 0) or (len(covered_neg) != 0):
            print("covered pos:")
            print(covered_pos)
            print("covered neg:")
            print(covered_neg)
        if len(covered_neg) > 0:
            return 0
        else:
            return len(covered_pos)
        """


    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        if eval > 0:
            return True
        else:
            return False

    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause],
                           hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]
        #  print("exps with body literals: " + str(exps))                                           # <-----------------
        # check if every clause has solutions
        exps = [(cl, self._solver.has_solution(*cl.get_body().get_literals())) for cl in exps]
        new_exps = []
        #  print("exps with a solution: " + str(exps))                                              # <-----------------
        for ind in range(len(exps)):
            if exps[ind][1]:
                # keep it if it has solutions
                new_exps.append(exps[ind][0])
            else:
                # remove from hypothesis space if it does not
                hypothesis_space.remove(exps[ind][0])
        return new_exps


if __name__ == '__main__':
    # defining Constants
    list_functor = Functor("[|]", 2)
    em = List([])  # the empty list
    sp = c_const("\' \'")  # the space character
    # defining Predicates
    take_word = c_pred("take_word", 3)
    second_word = c_pred("second_word", 2)
    head = c_pred("head", 2)
    tail = c_pred("tail", 2)

    # defining Variables
    X = c_var("X")
    Y = c_var("Y")
    X1 = c_var("X1")
    X2 = c_var("X2")
    Y1 = c_var("Y1")
    Y2 = c_var("Y2")
    Z = c_var("Z")
    g_ = c_const("\'g\'")
    i_ = c_const("\'i\'")
    e_ = c_const("\'e\'")
    l_ = c_const("\'l\'")
    n_ = c_const("\'n\'")
    d_ = c_const("\'d\'")
    Head = c_var("Head")
    Tail = c_var("Tail")
    Head1 = c_var("Head1")
    Head2 = c_var("Head2")
    Tail1 = c_var("Tail1")
    Tail2 = c_var("Tail2")

    # defining Clauses
    headClause = head(Pair(X, Y), X)
    tailClause = tail(Pair(X, Y), Y)
    """
    Below should be equivalent to what the folowing is in SWI Prolog:
    take_word([], [], 1)                  :- true
    take_word([' ', Tail ], [], 1)        :- true
    take_word([X  | Tail1], [X|Tail2], 1) :- take_word(Tail1, Tail2, 1)
    take_word([' '| Tail1], Tail2,     X) :- take_word(Tail1, Tail2, X-1)
    take_word([ _ | Tail1], Tail2,     X) :- take_word(Tail1, Tail2, X)

    This however contructing a list as a pair, of which the second element is itself a pair
    e.g., [a, b, c] becomes Pair(a, Pair(b, Pair(c, em))), with em representing an empty element

    """
    """
    case_0 = take_word(em, em, 1)
    case_1 = take_word(X, em, 1) <= head(X, sp)
    case_2 = take_word(X1, X2, 1) <= head(X1, X), head(X2, X), tail(X1, Y1), tail(X2, Y2), take_word(Y1, Y2, 1)
    case_3 = take_word(X1, Y2, 2) <= head(X1, sp), tail(X1, Y1), take_word(Y1, Y2, 1)
    case_4 = take_word(X1, Y2, 2) <= tail(X1, Y1), take_word(Y1, Y2, 2)
    case_5 = take_word(X1, Y2, 3) <= head(X1, sp), tail(X1, Y1), take_word(Y1, Y2, 2)
    case_6 = take_word(X1, Y2, 3) <= tail(X1, Y1), take_word(Y1, Y2, 3)
    """
    # specify the background knowledge
    background = Knowledge()
    # positive examples
    s_in = List(["g", "i", "e", "l", sp, "i", "n", "d", "e", sp, "k", "e", "u"])
    s_out = List(["i", "n", "d", "e"])

    ig_1 = List([g_, i_, e_, l_])
    ug_1 = g_
    ig_2 = List([l_, e_, i_])
    ug_2 = l_
    if_1 = List([g_, i_, e_, l_])
    uf_1 = i_
    if_2 = List([l_, e_, i_])
    uf_2 = e_
    pos = {second_word(ig_1, ug_1), second_word(ig_2, ug_2)}

    # negative examples
    neg = {second_word(if_1, uf_1), second_word(if_2, uf_2)}

    task = Task(positive_examples=pos, negative_examples=neg)

    # create Prolog instance and add general knowledge
    prolog = SWIProlog()
    prolog.assertz(headClause)
    prolog.assertz(tailClause)

    learner = SimpleBreadthFirstLearner(prolog, max_body_literals=2)

    # create the hypothesis space           lambda x: plain_extension(x, take_word, connected_clauses=True),
    hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, head, connected_clauses=True),
                                            lambda x: plain_extension(x, tail, connected_clauses=True)],
                                head_constructor=second_word,
                                expansion_hooks_reject=[lambda x, y: has_duplicated_literal(x, y)],
                                recursive_procedures=True)
                                                        # lambda x, y: has_singleton_vars(x, y),
    program = learner.learn(task, background, hs)

    print(program)




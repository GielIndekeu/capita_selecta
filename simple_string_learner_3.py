import typing
from abc import ABC, abstractmethod
from timeout import timeout

from orderedset import OrderedSet
from pylo.language.commons import Functor

from loreleai.language.lp import c_pred, Clause, Procedure, Atom, c_var, c_const, Body, List, Pair, list_func, Not
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

    # overrides learn of the TemplateLearner
    def _execute_program(self, clause: Clause, examples: Task) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge

        Returns a set of atoms that the clause covers
        """
        #  print("_execute_program(self, clause) => => trying clause: " + str(clause))
        if len(clause.get_body().get_literals()) == 0:
            return []
        if not self._solver.has_solution(*clause.get_body().get_literals()):
            return []
        self._solver.assertz(clause=clause)
        pos, _ = examples.get_examples()
        covered = []
        for example in pos:
            if self._solver.query(example):
                covered.append(example)
        self._solver.retract(clause=clause)     # <-- just added
        return covered

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        if isinstance(candidates, Clause):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)

    @timeout(5)
    def _has_solution(self, query):
        has_solution = self._solver.has_solution(query)
        return has_solution

    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        """
        This function differs from the origional in that it does not check which examples are covered, but checks
        instead rather or not the clause has a solution
        """
        print("evaluating: %60s" % str(clause), end='')
        if len(clause.get_body().get_literals()) == 0:
            print(" -> no body literals")
            return 0
        has_solution = False
        try:
            has_solution = self._has_solution(*clause.get_body().get_literals())
        except Exception:
            print(" -> timeout on 'has_solution', no solution presumed")
            return 0
        # has_solution = self._solver.has_solution(*clause.get_body().get_literals())
        if not has_solution:
            print(" -> has no solution")
            return 0
        self._solver.assertz(clause=clause)
        pos, neg = examples.get_examples()
        for example in neg:
            if self._solver.query(example):
                self._solver.retract(clause=clause)
                print(" -> covers a negative example: " + str(example))
                return 0
        covered = 0
        for example in pos:
            if self._solver.query(example):
                covered += 1
        print(" -> " + str(covered) + " examples covered")
        self._solver.retract(clause=clause)
        return covered

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

    # overrides learn of the TemplateLearner
    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: TopDownHypothesisSpace):
        """
        General learning loop
        """

        self._assert_knowledge(knowledge)
        final_program = []
        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()
        while len(final_program) == 0 or len(pos) > 0:
            # learn na single clause
            cl = self._learn_one_clause(examples_to_use, hypothesis_space)
            final_program.append(cl)
            # update covered positive examples
            covered = self._execute_program(cl, examples)
            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)
            examples_to_use = Task(pos, neg)
        return final_program


if __name__ == '__main__':
    # defining Constants
    list_functor = Functor("[|]", 2)
    em = List([])  # the empty list
    sp = c_const("\' \'")  # the space character
    # defining Predicates
    take_word = c_pred("take_word", 3)
    split_on_first_word = c_pred("split_on_first_word", 3)
    split_on_second_word = c_pred("split_on_second_word", 3)
    second_word = c_pred("second_word", 2)
    head = c_pred("head", 2)
    tail = c_pred("tail", 2)
    equal = c_pred("equal", 2)

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
    k_ = c_const("\'k\'")
    u_ = c_const("\'u\'")
    j_ = c_const("\'j\'")
    a_ = c_const("\'a\'")
    v_ = c_const("\'v\'")
    o_ = c_const("\'o\'")
    s_ = c_const("\'s\'")
    Head = c_var("Head")
    Tail = c_var("Tail")
    Head1 = c_var("Head1")
    Head2 = c_var("Head2")
    Tail1 = c_var("Tail1")
    Tail2 = c_var("Tail2")

    # defining Clauses
    headClause = head(Pair(X, Y), X)
    tailClause = tail(Pair(X, Y), Y)
    equalClause = equal(X, X)

    case_0 = split_on_first_word(em, em, em)
    case_1 = split_on_first_word(Pair(X, em), List([X]), em)
    case_2 = (split_on_first_word(Pair(X, Y1), List([X]), Y) <= head(Y1, sp) & tail(Y1, Y))
    case_3 = (split_on_first_word(Pair(X, Y1), Pair(X, Y2), Y) <= split_on_first_word(Y1, Y2, Y) & Not(equal(X, sp)))
    case_b0 = split_on_second_word(em, em, em)
    case_b1 = (split_on_second_word(X, Y, Z) <= split_on_first_word(X, Y1, Y2) & split_on_first_word(Y2, Y, Z))
    """
    case_2 = Clause(Atom(split_on_first_word, [Pair(X, Y1), List([X]), Y]),
                    Body(Atom(head, [Y1, sp]), Atom(tail, [Y1, Y])))
    case_3 = Clause(Atom(split_on_first_word, [Pair(X, Y1), Pair(X, Y2), Y]),
                    Body(Atom(split_on_first_word, [Y1, Y2, Y])))

    case_b0 = split_on_second_word(em, em, em)
    case_b1 = Clause(Atom(split_on_second_word, [X, Y, Z]),
                     Body(Atom(split_on_first_word, [X, Y1, Y2]), Atom(split_on_first_word, [Y2, Y, Z])))
    """
    """
    case_0 = take_word(em, em, 1)
    case_1 = (take_word(X, em, 1) <= head(X, sp))
    case_2 = (take_word(X1, X2, 1) <= head(X1, X) & head(X2, X) & tail(X1, Y1) & tail(X2, Y2) & take_word(Y1, Y2, 1))
    case_3 = (take_word(X1, Y2, 2) <= head(X1, sp) & tail(X1, Y1) & take_word(Y1, Y2, 1))
    case_4 = (take_word(X1, Y2, 2) <= tail(X1, Y1) & take_word(Y1, Y2, 2))
    case_5 = (take_word(X1, Y2, 3) <= head(X1, sp) & tail(X1, Y1) & take_word(Y1, Y2, 2))
    case_6 = (take_word(X1, Y2, 3) <= tail(X1, Y1) & take_word(Y1, Y2, 3))
    """
    # specify the background knowledge
    background = Knowledge()
    # positive examples
    s_in1 = List([g_, i_, e_, l_, sp, i_, n_, d_, e_, sp, k_, e_, u_])
    # s_out1 = List([g_, i_, e_, l_])
    s_out1 = List([i_, n_, d_, e_])
    s_in2 = List([j_, a_, n_, sp, v_, o_, s_, s_, sp, g_, i_])
    # s_out2 = List([j_, a_, n_])
    s_out2 = List([v_, o_, s_, s_])
    s_in3 = List([g_, i_, e_, l_, sp, i_, n_, d_, e_, sp, k_, e_, u_])
    s_out3 = List([i_, e_, l_])
    s_in4 = List([g_, i_, e_, l_, sp, i_, n_, d_, e_, sp, k_, e_, u_])
    s_out4 = List([k_, e_, u_])

    ig_1 = List([g_, i_, e_, l_])
    ug_1 = i_
    ig_2 = List([l_, e_, i_])
    ug_2 = e_
    if_1 = List([g_, i_, e_, l_])
    uf_1 = g_
    if_2 = List([l_, e_, i_])
    uf_2 = List([l_, e_, i_])
    pos = {second_word(s_in1, s_out1), second_word(s_in2, s_out2)}

    # negative examples
    neg = {second_word(s_in3, s_out3), second_word(s_in4, s_out4)}

    task = Task(positive_examples=pos, negative_examples=neg)

    # create Prolog instance and add general knowledge
    prolog = SWIProlog()
    prolog.assertz(headClause)
    prolog.assertz(tailClause)
    prolog.assertz(equalClause)
    prolog.assertz(case_0)
    prolog.assertz(case_1)
    prolog.assertz(case_2)
    prolog.assertz(case_3)

    prolog.assertz(case_b0)
    prolog.assertz(case_b1)
    """
    cl = (second_word(X, Y) <= (split_on_first_word(X, Y1, Y2) & split_on_first_word(Y2, Y, Z)))
    prolog.assertz(cl)
    print(prolog.query(second_word(s_in4, s_out4)))"""
    # prolog.assertz(case_2)
    # prolog.assertz(case_3)
    # prolog.assertz(case_4)
    # prolog.assertz(case_5)
    # prolog.assertz(case_6)

    learner = SimpleBreadthFirstLearner(prolog, max_body_literals=2)

    # create the hypothesis space                   lambda x: plain_extension(x, head, connected_clauses=True),
    #                                               lambda x: plain_extension(x, tail, connected_clauses=True),
    #                                               lambda x: plain_extension(x, split_on_second_word, connected_clauses=True),
    """
    hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, split_on_first_word, connected_clauses=True),
                                            lambda x: plain_extension(x, split_on_second_word, connected_clauses=True)],
                                head_constructor=second_word,
                                expansion_hooks_reject=[lambda x, y: has_duplicated_literal(x, y)],
                                recursive_procedures=False)
    # lambda x, y: has_singleton_vars(x, y),
    """
    hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, split_on_first_word, connected_clauses=True)],
                                head_constructor=second_word,
                                expansion_hooks_reject=[],
                                recursive_procedures=True)

    program = learner.learn(task, background, hs)

    print(program)

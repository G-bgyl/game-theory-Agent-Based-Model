import numpy as np
import munkres
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import deque


def make_id_generator():
    i = 0
    while True:
        yield i
        i += 1

REVIEWER_RESEARCH_QUALITY_CUTOFF = 0.5

id_generator = make_id_generator()


actuarial_death_table = {
    20: 0.001019,
    21: 0.001151,
    22: 0.001252,
    23: 0.001309,
    24: 0.001335,
    25: 0.001349,
    26: 0.001369,
    27: 0.001391,
    28: 0.001422,
    29: 0.001459,
    30: 0.001498,
    31: 0.001536,
    32: 0.001576,
    33: 0.001616,
    34: 0.001661,
    35: 0.001716,
    36: 0.001782,
    37: 0.001854,
    38: 0.001931,
    39: 0.002018,
    40: 0.002123,
    41: 0.002252,
    42: 0.002413,
    43: 0.002611,
    44: 0.002845,
    45: 0.003109,
    46: 0.003402,
    47: 0.003736,
    48: 0.004987,
    49: 0.004533,
    50: 0.004987,
    51: 0.005473,
    52: 0.005997,
    53: 0.006560,
    54: 0.007159,
    55: 0.007803,
    56: 0.008480,
    57: 0.009170,
    58: 0.009863,
    59: 0.010572,
    60: 0.011354,
    61: 0.012202,
    62: 0.013061,
    63: 0.013920,
    64: 0.014819,
    65: 0.015826,
    66: 0.016986,
    67: 0.018295,
    68: 0.019776,
    69: 0.021448,
    70: 0.023380,
    71: 0.025549,
    72: 0.027885,
    73: 0.030374,
    74: 0.033099,
    75: 0.036254,
    76: 0.039882,
    77: 0.043879,
    78: 0.048256,
    79: 0.053123,
    80: 0.058711
}

research_death_table = {key: value*10 for key, value in actuarial_death_table.items()}


class ModelSingleton(object):

    def __init__(self):
        self.AUTHORS = dict()
        self.PAPERS = dict()
        self.CONFERENCES = dict()
        self.REVIEWERS = dict()

    def reset(self):
        self.AUTHORS = dict()
        self.PAPERS = dict()
        self.CONFERENCES = dict()
        self.REVIEWERS = dict()

S = ModelSingleton()


QUALITY_CUTOFF, TOP_PERCENT = 0, 1

class Conference(object):

    def __init__(self, reviewer_quality_cutoff, research_area, research_breadth, num_papers_per_reviewer, quality_cutoff_percent, top_percent, conference_type=QUALITY_CUTOFF):
        self.id = next(id_generator)
        S.CONFERENCES[self.id] = self
        self.reviewer_quality_cutoff = reviewer_quality_cutoff
        self.num_papers_per_reviewer = num_papers_per_reviewer
        self.quality_cutoff_percent = quality_cutoff_percent

        self.conference_type = conference_type
        if conference_type == QUALITY_CUTOFF:
            self.quality_cutoff = self.quality_cutoff_percent
        else:
            self.top_percent = top_percent

        self.research_area = research_area
        self.research_breadth = research_breadth
        self.paper_feedbacks = deque(maxlen=10)
        self.open_new_cycle()

    def  get_quality(self):
        if self.conference_type == QUALITY_CUTOFF:
            return self.quality_cutoff
        else:
            accepted_feedbacks = [feedback for feedback in self.paper_feedbacks if feedback.paper_status == 'accept']
            average_quality = np.mean([feedback.average_submission_signal for feedback in accepted_feedbacks])
            # print('average_quality', average_quality)
            if np.isnan(average_quality):
                return 1.0
            else:
                return average_quality




    def open_new_cycle(self):
        self.submitted_papers = {}
        self.registered_reviewers = {}
        self.registered_authors = {}
        self.reviewer_orderings = {}


    def ingest_reviewer_ordering(self, reviewer, ordering):
        self.reviewer_orderings[reviewer.id] = ordering


    def register_reviewer(self, reviewer_id):
        reviewer = S.REVIEWERS[reviewer_id]
        is_reviewer = reviewer.parent.is_reviewer(self.id)
        if is_reviewer:
            self.registered_reviewers[reviewer_id] = S.REVIEWERS[reviewer_id]
        return is_reviewer



    def assign_papers_to_reviewers(self):
        num_papers_per_reviewer = self.num_papers_per_reviewer
        reviewer_dict = self.registered_reviewers
        submitted_papers = self.submitted_papers

        original_num_reviewer_slots = num_papers_per_reviewer*len(reviewer_dict)
        num_slots = max(len(submitted_papers), original_num_reviewer_slots)
        slot_to_reviewer_id = dict()
        slot_to_paper_id = dict()
        reviewer_id_list = list(reviewer_dict.keys())
        paper_id_list = list(submitted_papers.keys())
        # assign slots
        cost_matrix = np.zeros((num_slots, num_slots))

        # if there are no papers submitted.
        if len(submitted_papers) == 0:
            return {reviewer_id: [] for reviewer_id in reviewer_dict}

        # if there are no reviewers
        if len(reviewer_dict) == 0:
            return {}


        for i in range(num_slots):
            slot_to_reviewer_id[i] = reviewer_id_list[i % len(reviewer_dict)]
        for i in range(num_slots):
            slot_to_paper_id[i] = paper_id_list[i % len(submitted_papers)]
        # build the cost matrix
        for y in range(num_slots):
            for x in range(num_slots):
                reviewer_id = slot_to_reviewer_id[y]
                paper_id = slot_to_paper_id[x]
                reviewer = reviewer_dict[reviewer_id]
                paper = submitted_papers[paper_id]
                overlap = reviewer.get_paper_research_overlap_score(paper)
                cost = -overlap
                cost_matrix[y, x] = cost

        indices = munkres.Munkres().compute(cost_matrix)
        reviewer_id_to_assigned_paper_ids = dict()
        for y, x in indices:
            reviewer_id = slot_to_reviewer_id[y]
            paper_id = slot_to_paper_id[x]
            if reviewer_id not in reviewer_id_to_assigned_paper_ids:
                reviewer_id_to_assigned_paper_ids[reviewer_id] = [paper_id]
            else:
                reviewer_id_to_assigned_paper_ids[reviewer_id].append(paper_id)
            assigned_papers = reviewer_id_to_assigned_paper_ids[reviewer_id]
            # remove duplicates and enforce that no reviewers are assigned more papers than they agreed to.
            reviewer_id_to_assigned_paper_ids[reviewer_id] = list(set(assigned_papers))[:num_papers_per_reviewer]
        return reviewer_id_to_assigned_paper_ids


    def submit_paper(self, author_id, paper):
        self.registered_authors[author_id] = S.AUTHORS[author_id]
        self.submitted_papers[paper.id] = paper

    def update_quality_cutoff(self):
        # what is a submission signal?
        #mean_signal = np.mean(self.all_submission_signals)
        #if np.isnan(mean_signal):
        #    self.quality_cutoff = 0.0
        #else:
        #    self.quality_cutoff = self.quality_cutoff_percent * np.mean(self.all_submission_signals)
        self.quality_cutoff = self.quality_cutoff_percent

    def accept_or_reject_papers(self, paper_reviews):
        print('average reviewer quality', np.mean([r.reviewer_quality for r in self.registered_reviewers.values()]))
        print('num reviewers', len(self.registered_reviewers))
        incomplete_feedback = {}
        for paper_id, reviews in paper_reviews.items():
            if reviews == []:
                feedback = PaperConferenceFeedback(self.id, 'unreviewed', None, [])
            else:
                average_sub_signal = np.mean([review.submission_signal for review in reviews])
                feedback = PaperConferenceFeedback(self.id, None, average_sub_signal, reviews)
            incomplete_feedback[paper_id] = feedback
        if self.conference_type == QUALITY_CUTOFF:
            for feedback in incomplete_feedback.values():
                if feedback.paper_status == 'unreviewed':
                    continue
                feedback.paper_status = 'accept' if feedback.average_submission_signal >= self.get_quality() else 'reject'
                self.paper_feedbacks.append(feedback)
        elif self.conference_type == TOP_PERCENT:
            incomplete_feedback_values_no_unreviewed = [x for x in incomplete_feedback.values() if x.paper_status != 'unreviewed']
            sorted_values = sorted(incomplete_feedback_values_no_unreviewed, key=lambda x: x.average_submission_signal)
            N = len(sorted_values)
            cutoff = (1 - self.top_percent) * N
            good_submissions = sorted_values[int(cutoff):]
            bad_submissions = sorted_values[:int(cutoff)]
            for feedback in good_submissions:
                if feedback.paper_status == 'unreviewed':
                    continue
                feedback.paper_status = 'accept'
                self.paper_feedbacks.append(feedback)
            for feedback in bad_submissions:
                if feedback.paper_status == 'unreviewed':
                    continue
                feedback.paper_status = 'reject'
                self.paper_feedbacks.append(feedback)
        else:
            raise Exception('Unknown Conference Type')

        return incomplete_feedback



    def accept_or_reject_paper(self, paper_id, reviews):
        if reviews == []:
            return PaperConferenceFeedback(self.id, 'unreviewed', None, [])
        average_sub_signal = np.mean([review.submission_signal - self.quality_cutoff for review in reviews])
        self.all_submission_signals.append(average_sub_signal)
        paper_status = 'accept' if average_sub_signal >= 0 else 'reject'

        feedback = PaperConferenceFeedback(self.id, paper_status, average_sub_signal, reviews)
        self.paper_feedbacks.append(feedback)
        return feedback

    def send_feedback_to_author(self, paper_id, paper_feedback):
        author = self.registered_authors[self.submitted_papers[paper_id].author_id]
        author.receive_paper_feedback(paper_id, paper_feedback)


class Paper(object):

    def __init__(self, author_id, paper_area, paper_quality, conference_id):
        self.id = next(id_generator)
        S.PAPERS[self.id] = self
        self.author_id = author_id
        self.paper_area = paper_area
        self.paper_quality = paper_quality
        self.conference_id = conference_id
        self.feedback_history = {}

    def store_feedback(self, cycle, feedback):
        self.feedback_history[cycle] = feedback


class PaperReviewFeedback(object):

    def __init__(self, submission_signal, feedback_quality):
        self.submission_signal = submission_signal
        self.feedback_quality = feedback_quality

class PaperConferenceFeedback(object):

    def __init__(self, conference_id, paper_status, average_submission_signal, review_feedback_list):
        # 0 is horrible, 1 is amazing.
        self.conference_id = conference_id
        self.paper_status = paper_status
        self.average_submission_signal = average_submission_signal
        self.review_feedback_list = review_feedback_list




class Author(object):

    def __init__(self, average_num_papers_poisson_variable, base_research_quality, research_area, research_breadth, effort_level, publishing_pressure):
        self.id = next(id_generator)
        S.AUTHORS[self.id] = self
        self.average_num_paper_poisson_variable = average_num_papers_poisson_variable
        self.base_research_quality = base_research_quality
        self.effort_level = effort_level
        self.publishing_pressure = publishing_pressure
        self.research_area = research_area
        self.research_breadth = research_breadth
        self.is_dead = False

        self.reviewer_module = Reviewer(self, np.random.uniform(0, 1), research_area, research_breadth, effort_level)

        # mapping between conference names and how an author feels about that particular conference.
        # authors can become unhappy with conferences when they feel "wronged" by the review process.
        self.conference_happiness = {}
        self.pending_papers = {}
        self.research_quality = self.base_research_quality
        self.age = np.random.randint(20, 50)


    def is_reviewer(self, conf_id):
        return self.reviewer_module.reviewer_quality >= S.CONFERENCES[conf_id].get_quality()

    def initialize_conference_happiness(self, conferences_list):
        for conference in conferences_list:
            conference_id = conference.id
            S.CONFERENCES[conference_id] = conference
            self.conference_happiness[conference_id] = 0.5

    def get_conference_overlap_score(self, conf_id, paper_area):
        conf_research_area = S.CONFERENCES[conf_id].research_area
        conf_research_breadth = S.CONFERENCES[conf_id].research_breadth
        Z = norm.pdf(conf_research_area, conf_research_area, conf_research_breadth)
        return norm.pdf(paper_area, conf_research_area, conf_research_breadth) / Z

    def assess_conference(self, conf_id, paper_area, conference_happiness, paper_quality):
        conference = S.CONFERENCES[conf_id]
        overlap_score = self.get_conference_overlap_score(conf_id, paper_area)
        conference_quality = conference.get_quality()
        quality_disparity = 1-np.abs(conference_quality - paper_quality)
        # if disparity is large, less likely to submit.
        a, b = 0.5, 0.5
        return a*(conference_happiness * overlap_score) + b*quality_disparity

    def produce_paper(self):
        paper_area = np.clip(np.random.normal(self.research_area, self.research_breadth), 0, 1)
        paper_quality = np.clip(np.random.normal(self.research_quality, 0.1), 0, 1)
        # isolate the conference happinesses, sorted by conference id index.
        happiness_times_overlap_items = [(conf_id, self.assess_conference(conf_id, paper_area, happiness, paper_quality))
                                         for (conf_id, happiness) in self.conference_happiness.items()]
        conference_list = np.array([y for x, y in sorted(happiness_times_overlap_items, key=lambda x: x[0])])
        # todo figure out why conference happiness is so frequently 50 - 50
        temp = 10.0
        # big temp --> large difference
        conference_probs = np.exp(conference_list * temp) / np.sum(np.exp(conference_list * temp))
        conference_id = np.random.choice(sorted(self.conference_happiness.keys()), p=conference_probs)

        paper_id = next(id_generator)
        paper = Paper(self.id, paper_area, paper_quality, conference_id)
        return paper

    def produce_papers(self):
        if not self.is_dead:
            num_papers = np.random.poisson(lam=self.average_num_paper_poisson_variable)
            produced_papers = [self.produce_paper() for _ in range(num_papers)]
            for paper in produced_papers:
                self.pending_papers[paper.id] = paper
            all_papers = produced_papers + list(self.pending_papers.values())
            return all_papers
        else:
            return []


    def submit_papers_to_conferences(self, papers):
        for paper in papers:
            S.CONFERENCES[paper.conference_id].submit_paper(paper.author_id, paper)

    def receive_paper_feedback(self, paper_id, paper_feedback):
        # increase research quality by some amount based on feedback quality and maybe author age?
        # decide based upon the quality of the paper whether it should be resubmitted

        # if the paper got in remove it from pending papers.
        paper = self.pending_papers[paper_id]
        conference = S.CONFERENCES[paper.conference_id]
        if paper_feedback.paper_status == 'unreviewed':
            #self.conference_happiness[conference.id] = 0.
            del self.pending_papers[paper_id]
        elif paper_feedback.paper_status == 'accept':
            del self.pending_papers[paper_id]
        # paper is too bad to consider resubmission
        else:
            del self.pending_papers[paper_id]
        #elif paper_feedback.average_submission_signal < -0.25:
        #    del self.pending_papers[paper_id]


        # increase research quality
        # assumes that it takes 20 years to become a master research assuming optimal feedback.
        #max_feedback_quality  = np.max([r.feedback_quality for r in paper_feedback.review_feedback_list])
        #perc = max_feedback_quality * (1./20)
        #self.research_quality += perc
        #self.research_quality = np.clip(self.research_quality, 0, 1)

        # update happiness
        average_feedback_quality = np.mean([r.feedback_quality for r in paper_feedback.review_feedback_list])
        #print('avg feedback qual', average_feedback_quality, 'avg_submission_signal', paper_feedback.average_submission_signal)

        feedback_disparity = np.mean([np.abs(r.submission_signal - paper.paper_quality) for r in paper_feedback.review_feedback_list])
        # print('feedback_info', paper.conference_id, feedback_disparity, average_feedback_quality)
        happiness_update = (average_feedback_quality/100.) - (feedback_disparity / 2.)
        perc = happiness_update / 5.
        conf_hap = self.conference_happiness[conference.id]
        conf_hap += perc
        conf_hap = np.clip(conf_hap, 0, 1)

        self.conference_happiness[conference.id] = conf_hap








class Reviewer(object):

    def __init__(self, parent_author_object, reviewer_quality, research_area, research_breadth, effort_level):
        self.parent = parent_author_object
        self.id = next(id_generator)
        S.REVIEWERS[self.id] = self
        self.reviewer_quality = reviewer_quality
        self.research_area = research_area
        self.min_quality = self.reviewer_quality - 0.5
        self.research_breadth = research_breadth
        self.effort_level = effort_level

        self.reset_assessed_conference_quality()

    def reset_assessed_conference_quality(self):
        self.assessed_conference_quality = {conf_id: [] for conf_id in S.CONFERENCES}


    def register_for_conferences(self):
        if self.parent.is_dead:
            return
        # get a sorted list of conferences.
        conference_list = np.array([np.mean(y) for x, y in sorted(self.assessed_conference_quality.items(), key=lambda x: x[0])])
        temp = 1.0
        conference_probs = np.exp(conference_list * temp) / np.sum(np.exp(conference_list * temp))
        # keys, paired with conference_probs
        keys_probs = zip(sorted(self.assessed_conference_quality.keys()), conference_probs)
        # remove sorting determinism
        keys_probs = sorted(keys_probs, key=lambda x: np.random.uniform())
        conference_ordering = [i for (i, x) in sorted(keys_probs, key=lambda x: -x[1])]
        # TODO make the max number of papers a variables for the reviewer
        number_of_papers = int(20*self.effort_level)
        num_papers_left = number_of_papers
        for conf_id in conference_ordering:
            conf = S.CONFERENCES[conf_id]
            #print(conf.get_quality())
            #if conf.reviewer_quality_cutoff < self.min_quality:
            #    continue
            if conf.num_papers_per_reviewer <= num_papers_left:
                registered_successfully = conf.register_reviewer(self.id)
                if registered_successfully:
                    num_papers_left -= conf.num_papers_per_reviewer


    def get_paper_research_overlap_score(self, paper):
        area = paper.paper_area
        Z = norm.pdf(self.research_area, self.research_area, self.research_breadth)
        return max(norm.pdf(area, self.research_area, self.research_breadth) / Z, 0.001)

    def get_research_overlap_ordering(self, papers):
        paper_scores = [self.get_paper_research_overlap_score(paper) for paper in papers]
        ordering = [i for i, score in sorted(enumerate(paper_scores), key=lambda x: -x[1])]
        return ordering

    def review_paper(self, paper):
        # reviews should be noisy depending on how far the research area is away from the "breadth" of the reviewer.
        overlap_score = self.get_paper_research_overlap_score(paper)
        #review_stddev = min(0.01 / overlap_score, 1.0)
        review_stddev = 1. - self.reviewer_quality
        #review_stddev = 1. - overlap_score
        #print('overlap_score', overlap_score)
        # TODO why is the conference quality > than the conference_id
        submission_signal = np.clip(np.random.normal(paper.paper_quality, review_stddev), 0, 1)
        feedback_quality = overlap_score * self.effort_level
        self.assessed_conference_quality[paper.conference_id].append(submission_signal)
        return PaperReviewFeedback(submission_signal, feedback_quality)


class AgentSystem(object):

    def __init__(self, num_years, num_authors, conferences_list):
        self.num_years = num_years
        self.num_authors = num_authors

        self.create_authors(self.num_authors)
        self.cycle_number = 0

    def create_authors(self, num_authors):
        for i in range(num_authors):
            average_num_papers_poisson_variable = np.random.uniform(0.25, 0.5)
            base_research_quality = np.random.uniform(0, 1)
            # treat research area as a spectrum.
            research_area = np.random.uniform(0, 1)
            effort_level = np.random.uniform(0, 1)
            publishing_pressure = np.random.uniform(0, 1)
            research_breadth = np.random.uniform(0, 0.2)
            # research quality can improve by (1) submitting more research (2) by receiving meaningful input from reviewers
            author = Author(average_num_papers_poisson_variable, base_research_quality, research_area, research_breadth, effort_level, publishing_pressure)
            author.initialize_conference_happiness(S.CONFERENCES.values())
            S.AUTHORS[id] = author

    #def create_reviewers(self):
    #    self.reviewers = {}
    #    for i in range(self.num_reviewers):
    #        reviewer_quality = np.random.uniform(0, 1)
    #        research_area = np.random.uniform(0, 1)
    #        research_breadth = np.random.uniform(0, 1)
    #        effort_level = np.random.uniform(0, 1)
    #        reviewer = Reviewer(reviewer_quality, research_area, research_breadth, effort_level)
    #        reviewer.initialize_conference_happiness(self.conferences.values())
    #        self.reviewers[reviewer.id] = reviewer

    def do_cycle(self):
        print('doing cycle', self.cycle_number)
        # for each conference
        #   authors produce papers
        #   papers get submitted to conferences
        #   reviewers register for conferences
        #   reviewers choose papers
        #   reviewers are assigned papers
        #   papers are either accepted or rejected
        #   authors decide whether or not to resubmit.
        #   conference happiness is updated for authors and reviewers.
        #   authors research quality is updated and old authors die.
        for conference_id, conference in S.CONFERENCES.items():
            conference.open_new_cycle()


        # aggregate papers by conference.
        for author_id, author in S.AUTHORS.items():

            papers = author.produce_papers()
            author.submit_papers_to_conferences(papers)
            author.reviewer_module.register_for_conferences()
            author.reviewer_module.reset_assessed_conference_quality()

        for conference_id, conference in S.CONFERENCES.items():
            # authors produce papers
            #qualified_reviewers = {id: author.reviewer_module for id, author in S.AUTHORS.items()
            #             if author.is_reviewer(conference_id)}
            #print('conference', conference_id)
            #for id, reviewer in qualified_reviewers.items():
            #    print('reviewer_quality', reviewer.parent.research_quality)

            # reviewers register for the conferences.
            #for reviewer_id, reviewer in qualified_reviewers.items():
            #    reviewer.register_for_conferences()

            registered_reviewers = conference.registered_reviewers
            # reviewers
            for reviewer_id, reviewer in registered_reviewers.items():
                paper_ordering = reviewer.get_research_overlap_ordering(conference.submitted_papers.values())
                conference.ingest_reviewer_ordering(reviewer, paper_ordering)
            # should be a dictionary mapping reviewer_ids to a list of papers_ids.
            assignment = conference.assign_papers_to_reviewers()
            # assign papers to reviewer and collect reviews.
            paper_reviews = {paper_id: [] for paper_id in conference.submitted_papers}
            for reviewer_id, assigned_papers in assignment.items():
                for paper_id in assigned_papers:
                    paper = S.PAPERS[paper_id]
                    review = registered_reviewers[reviewer_id].review_paper(paper)
                    paper_reviews[paper.id].append(review)

            for paper_id, paper_feedback in conference.accept_or_reject_papers(paper_reviews).items():
                S.PAPERS[paper_id].store_feedback(self.cycle_number, paper_feedback)
                conference.send_feedback_to_author(paper_id, paper_feedback)

            #for paper_id, reviews in paper_reviews.items():
            #    paper_feedback = conference.accept_or_reject_paper(paper_id, reviews)
            #    S.PAPERS[paper_id].store_feedback(self.cycle_number, paper_feedback)
            #    conference.send_feedback_to_author(paper_id, paper_feedback)

            # update the quality_cutoff of the conference according to the average submission signal.
            conference.update_quality_cutoff()

        # age authors and kill off old oness
        for author_id, author in S.AUTHORS.items():
            author.age += 1
            prob_research_death = research_death_table.get(author.age, 1.0)

            died = np.random.binomial(1, [prob_research_death])[0]
            if died == 1:
                author.is_dead = True

        # compute number of living authors
        living_count = 0
        for author_id, author in S.AUTHORS.items():
            living_count += (0 if author.is_dead else 1)

        # create new authors to replace the dead ones
        num_to_create = max(self.num_authors - living_count, 0)
        self.create_authors(num_to_create)

        stats = {
            'reviewer_quality_assessments': {id: self.get_reviewer_quality_assessments(conference)
                                             for id, conference in S.CONFERENCES.items()},
            'author_conference_happiness': {id: self.get_author_happiness(conference)
                                            for id, conference in S.CONFERENCES.items()},
            'num_reviewers': {id: self.get_number_registered_reviewers(conference)
                          for id, conference in S.CONFERENCES.items()},
            'num_submitted': {id: self.get_number_submitted_papers(conference)
                              for id, conference in S.CONFERENCES.items()},
            'percent_accepted': {id: self.get_number_accepted_papers(self.cycle_number, conference)
                              for id, conference in S.CONFERENCES.items()},
            'paper_true_quality': {id: self.get_average_paper_true_quality(conference)
                                   for id, conference in S.CONFERENCES.items()}
        }



        self.cycle_number += 1

        return stats

    def get_average_paper_true_quality(self, conference):
        return np.mean([x.paper_quality for x in conference.submitted_papers.values()])

    def get_reviewer_quality_assessments(self, conference):
        reviewers = conference.registered_reviewers.values()
        overall_quality_list = []
        for reviewer in reviewers:
            quality_list = reviewer.assessed_conference_quality[conference.id]
            quality = np.mean(quality_list)
            overall_quality_list.append(quality)
        return overall_quality_list

    def get_author_happiness(self, conference):
        authors = conference.registered_authors.values()
        return [author.conference_happiness[conference.id] for author in authors]

    def get_number_registered_reviewers(self, conference):
        return len(conference.registered_reviewers)

    def get_number_submitted_papers(self, conference):
        return len(conference.submitted_papers)

    def get_number_accepted_papers(self, cycle_number, conference):
        papers = conference.submitted_papers.values()
        return np.mean([paper.feedback_history[cycle_number].paper_status == 'accept' for paper in papers])

    def report_cycle_statistics(self):
        return {}

    def run_for_n_cycles(self, num_cycles):
        S.reset()
        all_cycle_statistics = []
        for i in range(num_cycles):
            statistics = self.do_cycle()
            all_cycle_statistics.append(statistics)
        return all_cycle_statistics



conf1 = Conference(reviewer_quality_cutoff=0.4,
                   research_area=0.1,
                   research_breadth=0.3,
                   num_papers_per_reviewer=5,
                   top_percent=0.40,
                   quality_cutoff_percent=0.6,
                   conference_type=TOP_PERCENT)

conf2 = Conference(reviewer_quality_cutoff=0.5,
                   research_area=0.1,
                   research_breadth=0.3,
                   num_papers_per_reviewer=5,
                   top_percent=0.60,
                   quality_cutoff_percent=0.5,
                   conference_type=QUALITY_CUTOFF)


print('conf1_id', conf1.id)

#conf2 = Conference(reviewer_quality_cutoff=0.0,
#                   research_area=0.1,
#                   research_breadth=0.3,
#                   num_papers_per_reviewer=3,
#                   top_percent=0.40,
#                   quality_cutoff_percent=0.6,
#                   conference_type=QUALITY_CUTOFF)
print('conf2_id', conf2.id)

agent_system = AgentSystem(num_years=1, num_authors=40, conferences_list=[conf1, conf2])
num_cycles = 30
all_stats = [agent_system.do_cycle() for i in range(num_cycles)]
conference_keys = sorted(S.CONFERENCES.keys())

all_happiness = {conf: [] for conf in conference_keys}

all_quality = {conf: [] for conf in conference_keys}
all_num_reviewers = {conf: [] for conf in conference_keys}
all_num_submitted = {conf: [] for conf in conference_keys}
all_percent_accepted = {conf: [] for conf in conference_keys}
all_paper_quality = {conf: [] for conf in conference_keys}

def plot_statistics(key, conf1, conf2, all_stats, take_mean=False):
    conf1_stats, conf2_stats = [], []
    for i in range(num_cycles):
        if take_mean:
            conf1_stat, conf2_stat = np.mean(all_stats[i][key][conf1]), np.mean(all_stats[i][key][conf2])
        else:
            conf1_stat, conf2_stat = all_stats[i][key][conf1], all_stats[i][key][conf2]

        conf1_stats.append(conf1_stat)
        conf2_stats.append(conf2_stat)
    f, ax = plt.subplots()
    ax.set_title(key)
    ax.plot(conf1_stats, color='red')
    ax.plot(conf2_stats, color='blue')
    f.savefig(key+'.png')

#conf1, conf2 = conference_keys

plot_statistics('author_conference_happiness', conf1.id, conf2.id, all_stats, take_mean=True)
plot_statistics('reviewer_quality_assessments', conf1.id, conf2.id, all_stats, take_mean=True)
plot_statistics('num_reviewers', conf1.id, conf2.id, all_stats)
plot_statistics('num_submitted', conf1.id, conf2.id, all_stats)
plot_statistics('percent_accepted', conf1.id, conf2.id, all_stats)
plot_statistics('paper_true_quality', conf1.id, conf2.id, all_stats)

for i in range(num_cycles):
    for conf in conference_keys:
        stats = all_stats[i]
        author_happiness = np.mean(stats['author_conference_happiness'][conf])
        reviewer_quality_assessment = np.mean(stats['reviewer_quality_assessments'][conf])
        num_reviewers = stats['num_reviewers'][conf]
        num_submitted = stats['num_submitted'][conf]
        percent_accepted = stats['percent_accepted'][conf]
        paper_quality = stats['paper_true_quality'][conf]
        all_paper_quality[conf].append(paper_quality)
        all_happiness[conf].append(author_happiness)
        all_quality[conf].append(reviewer_quality_assessment)
        all_num_reviewers[conf].append(num_reviewers)
        all_num_submitted[conf].append(num_submitted)
        all_percent_accepted[conf].append(percent_accepted)

print('all happiness:\n',all_happiness)

for conf in conference_keys:
    f, axs = plt.subplots(2,1)
    axs[0].set_title('Fractional Quantities')
    axs[0].plot(all_happiness[conf], color='blue')
    axs[0].plot(all_quality[conf], color='green')
    axs[0].plot(all_percent_accepted[conf], color='black')
    axs[0].plot(all_paper_quality[conf], color='purple')

    axs[1].set_title('Integer Quantities')
    axs[1].plot(all_num_reviewers[conf], color='red')
    axs[1].plot(all_num_submitted[conf], color='yellow')
    f.savefig('happiness_and_quality_plot_conf_%s.png' % conf)
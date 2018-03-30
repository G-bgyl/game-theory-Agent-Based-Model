import munkres
import numpy as np


def perform_matching(reviewer_dict, conference):
    num_papers_per_reviewer = conference.num_papers_per_reviewer
    submitted_papers = conference.submitted_papers
    reviewer_id_to_index = dict()
    index_to_reviewer_id = dict()
    reviewer_id_to_assigned_paper_ids = dict()
    index_to_paper_id = dict()

    cost_matrix = np.zeros((num_papers_per_reviewer * len(reviewer_dict)))
    for i, (reviewer_id, reviewer) in enumerate(reviewer_dict.items()):
        reviewer_id_to_assigned_paper_ids[reviewer_id] = []
        reviewer_id_to_index[reviewer_id] = i
        index_to_reviewer_id[i] = reviewer_id

        for n in range(num_papers_per_reviewer):
            for x, (paper_id, paper) in enumerate(submitted_papers.items()):
                index_to_paper_id[x] = paper_id
                y = i*num_papers_per_reviewer + n
                overlap = reviewer.get_paper_research_overlap_score(paper)
                cost = -overlap
                cost_matrix[y, x] = cost
    indices = munkres.Munkres().compute(cost_matrix)
    for reviewer_slot, paper_slot in indices:
        index = reviewer_slot // num_papers_per_reviewer
        reviewer_id = index_to_reviewer_id[index]
        paper_id = index_to_paper_id[paper_slot]
        reviewer_id_to_assigned_paper_ids[reviewer_id].append(paper_id)
    return reviewer_id_to_assigned_paper_ids









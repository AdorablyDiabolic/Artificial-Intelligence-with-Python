import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probabilities = {}
    pages = list(corpus.keys())
    links = corpus[page]

    if not links:  # If no outgoing links, treat it as if there are links to all pages
        links = set(pages)

    # Probability of choosing among all pages (random jump)
    for p in pages:
        probabilities[p] = (1 - damping_factor) / len(pages)

    # Additional probability of choosing among the linked pages
    if links:
        num_links = len(links)
        for link in links:
            probabilities[link] += damping_factor / num_links

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # Initialize page counts
    page_rank = {page: 0 for page in corpus.keys()}

    # Start from a random page
    current_page = random.choice(list(corpus.keys()))
    page_rank[current_page] += 1

    # Generate the rest of the samples
    for _ in range(1, n):
        current_probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(current_probabilities.keys()), 
                                      weights=current_probabilities.values(), k=1)[0]
        page_rank[current_page] += 1

    # Normalize the counts to probabilities
    total_samples = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_samples

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    ranks = {page: 1/N for page in corpus}
    new_ranks = ranks.copy()
    change = True

    # PageRank iteration until ranks converge within 0.001
    while change:
        change = False
        for page in corpus:
            new_rank = (1 - damping_factor) / N
            # Sum contributions from pages that link to the current page
            for potential_linker in corpus:
                if page in corpus[potential_linker] or not corpus[potential_linker]:
                    if corpus[potential_linker]:  # If the linker has links
                        links = len(corpus[potential_linker])
                    else:  # Treat no links as links to all pages
                        links = N
                    new_rank += damping_factor * (ranks[potential_linker] / links)
            # Check if change is within the tolerance
            if abs(new_ranks[page] - new_rank) >= 0.001:
                change = True
            new_ranks[page] = new_rank

        ranks = new_ranks.copy()

    return ranks


if __name__ == "__main__":
    main()

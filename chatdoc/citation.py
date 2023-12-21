from dataclasses import dataclass
from abc import abstractmethod
from typing import Any


@dataclass(frozen=True)
class BaseCitation:
    """
    A citation for a source in a document
    """

    source: str
    page: int

    @abstractmethod
    def format_citation_text(self):
        """
        Format the text from the citation
        """
        return f" - {self.source} on page {self.page}"


@dataclass(frozen=True)
class ProofCitation(BaseCitation):
    """
    A base citation with proof
    """

    proof: str

    def format_citation_text(self):
        """
        Format the text from the citation
        """
        return f" - {self.source} on page {self.page}; PROOF: {self.proof}"


Citation = BaseCitation | ProofCitation


@dataclass
class Citations:
    """
    A set of citations
    """

    citations: set[Citation]
    with_proof: bool

    def add_citation(self, source: str, page: int, proof: str):
        """
        Add a citation to the set of citations
        """
        citation: Citation = ProofCitation(source, page, proof) if self.with_proof else BaseCitation(source, page)
        self.citations.add(citation)

    def get_unique_citations(self, source_documents: list[Any]):
        """
        Get the unique citations from the source documents
        """
        for source_document in source_documents:
            source = source_document.metadata["source"]
            page = source_document.metadata["page"] + 1
            proof = source_document.page_content
            self.add_citation(source, page, proof)

    def print_citations(self):
        """
        Print the citations
        """
        for citation in self.citations:
            print(citation.format_citation_text(), flush=True)

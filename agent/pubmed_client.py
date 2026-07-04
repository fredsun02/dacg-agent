"""
PubMed API Client

Handles searching and fetching papers from PubMed using NCBI E-utilities.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class Paper:
    """Paper data structure"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    pub_year: int
    mesh_terms: List[str]


class PubMedClient:
    """PubMed API client with rate limiting and error handling"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 0.35,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ):
        """
        Initialize PubMed client.

        Args:
            api_key: NCBI API key (optional, allows higher rate limit)
            rate_limit: Seconds between requests
            base_url: E-utilities base URL
        """
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.base_url = base_url
        if rate_limit is None:
            rate_limit = 0.12 if self.api_key else 0.35
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.last_error = None
        self.last_error_time = None

    def _clear_error(self):
        self.last_error = None
        self.last_error_time = None

    def _set_error(self, context: str, error: Exception):
        self.last_error = f"{context}: {type(error).__name__}: {error}"
        self.last_error_time = time.time()

    def _wait_rate_limit(self):
        """Wait to respect rate limit"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 100,
        start: int = 0,
        sort: str = "relevance"
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.

        Args:
            query: Search query
            max_results: Maximum results to return
            start: Starting offset for pagination
            sort: Sort order ("relevance" or "date")

        Returns:
            List of PMIDs
        """
        self._wait_rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retstart": start,
            "retmode": "json",
            "sort": sort
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            self._clear_error()
            return pmids

        except requests.RequestException as e:
            self._set_error("search", e)
            print(f"PubMed search error: {e}")
            return []

    def search_count(self, query: str) -> int:
        """
        Return total hit count for a PubMed query.

        Args:
            query: Search query

        Returns:
            Total count of matching PMIDs.
        """
        self._wait_rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retstart": 0,
            "retmode": "json"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            count_str = data.get("esearchresult", {}).get("count", "0")
            self._clear_error()
            return int(count_str)

        except requests.RequestException as e:
            self._set_error("search_count", e)
            print(f"PubMed search_count error: {e}")
            return 0

    def fetch_papers(self, pmids: List[str]) -> List[Paper]:
        """
        Fetch paper details for given PMIDs.

        Args:
            pmids: List of PMIDs

        Returns:
            List of Paper objects
        """
        if not pmids:
            return []

        self._wait_rate_limit()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(
                f"{self.base_url}/efetch.fcgi",
                params=params,
                timeout=60
            )
            response.raise_for_status()

            papers = self._parse_xml(response.text)
            self._clear_error()
            return papers

        except requests.RequestException as e:
            self._set_error("fetch", e)
            print(f"PubMed fetch error: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> List[Paper]:
        """Parse PubMed XML response"""
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID", "")

                # Title
                title = article.findtext(".//ArticleTitle", "")

                # Abstract - concatenate all abstract parts
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    (part.text or "") for part in abstract_parts
                )

                # Authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.findtext("LastName", "")
                    first_name = author.findtext("ForeName", "")
                    if last_name:
                        authors.append(f"{last_name} {first_name}".strip())

                # Journal
                journal = article.findtext(".//Journal/Title", "")

                # Publication date
                pub_date = article.findtext(".//PubDate/Year", "")
                if not pub_date:
                    pub_date = article.findtext(".//PubDate/MedlineDate", "")[:4] if article.findtext(".//PubDate/MedlineDate") else ""

                try:
                    pub_year = int(pub_date[:4]) if pub_date else 0
                except ValueError:
                    pub_year = 0

                # MeSH terms
                mesh_terms = [
                    mesh.findtext("DescriptorName", "")
                    for mesh in article.findall(".//MeshHeading")
                    if mesh.findtext("DescriptorName")
                ]

                # Only keep papers with abstracts
                if pmid and abstract:
                    papers.append(Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        pub_date=pub_date,
                        pub_year=pub_year,
                        mesh_terms=mesh_terms
                    ))

        except ET.ParseError as e:
            print(f"XML parse error: {e}")

        return papers

    def search_and_fetch(
        self,
        query: str,
        batch_size: int = 5,
        max_batches: int = 20
    ) -> Generator[Tuple[int, List[Paper]], None, None]:
        """
        Generator: search and yield papers in batches.

        Args:
            query: Search query
            batch_size: Papers per batch
            max_batches: Maximum number of batches

        Yields:
            (batch_index, list of Papers)
        """
        # First get all PMIDs
        print(f"[PubMed] Searching: {query}")
        pmids = self.search(query, max_results=batch_size * max_batches)
        print(f"[PubMed] Found {len(pmids)} PMIDs")

        if not pmids:
            print("[PubMed] No results found")
            return

        # Yield in batches
        for batch_idx in range(0, len(pmids), batch_size):
            if batch_idx // batch_size >= max_batches:
                break

            batch_pmids = pmids[batch_idx:batch_idx + batch_size]
            papers = self.fetch_papers(batch_pmids)
            print(f"[PubMed] Batch {batch_idx // batch_size}: fetched {len(papers)} papers with abstracts")

            yield batch_idx // batch_size, papers

    def build_causal_query(
        self,
        head_entity: str,
        tail_entity: str,
        additional_terms: Optional[List[str]] = None
    ) -> str:
        """
        Build a PubMed query for causal relationship search.

        Args:
            head_entity: Treatment/intervention entity
            tail_entity: Condition/outcome entity
            additional_terms: Extra search terms

        Returns:
            Formatted PubMed query string
        """
        # Base query with entities
        query_parts = [f'("{head_entity}"[Title/Abstract])', f'("{tail_entity}"[Title/Abstract])']

        # Add causal keywords
        causal_keywords = [
            "treatment", "therapy", "effect", "efficacy",
            "cause", "prevent", "reduce", "increase",
            "associated", "correlation", "outcome"
        ]
        causal_part = " OR ".join(f'"{kw}"[Title/Abstract]' for kw in causal_keywords)
        query_parts.append(f"({causal_part})")

        # Add any additional terms
        if additional_terms:
            for term in additional_terms:
                query_parts.append(f'("{term}"[Title/Abstract])')

        # Combine with AND
        query = " AND ".join(query_parts)

        return query


# Test function
if __name__ == "__main__":
    client = PubMedClient()

    # Test search
    query = "aspirin cardiovascular disease prevention"
    print(f"Searching: {query}")

    for batch_idx, papers in client.search_and_fetch(query, batch_size=3, max_batches=2):
        print(f"\nBatch {batch_idx}: {len(papers)} papers")
        for p in papers[:2]:
            print(f"  - {p.pmid}: {p.title[:60]}...")
            print(f"    Year: {p.pub_year}, Abstract length: {len(p.abstract)}")

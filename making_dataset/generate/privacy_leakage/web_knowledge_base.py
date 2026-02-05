"""
Curated web knowledge base for 4-hop question generation.

Contains facts about real-world entities that can be used to create
Local → Web → Local → Web reasoning chains.
"""
from __future__ import annotations

from .schema import WebEntity, WebKnowledgeGraph


def build_web_knowledge_graph() -> WebKnowledgeGraph:
    """Build the web knowledge graph with curated entities."""
    graph = WebKnowledgeGraph()

    # Add all entity categories
    _add_technology_entities(graph)
    _add_company_entities(graph)
    _add_compliance_entities(graph)
    _add_certification_entities(graph)

    return graph


def _add_technology_entities(graph: WebKnowledgeGraph) -> None:
    """Add technology/software entities."""

    # Container Orchestration
    graph.add_entity(WebEntity(
        name="Kubernetes",
        entity_type="technology",
        facts={
            "developed_by": "Google",
            "donated_to": "CNCF",
            "release_year": "2014",
            "headquarters": "San Francisco",  # CNCF location
            "written_in": "Go",
        },
        ecosystem={
            "monitoring": ["Prometheus", "Grafana", "Datadog"],
            "service_mesh": ["Istio", "Linkerd", "Consul"],
            "ci_cd": ["Argo CD", "Flux", "Jenkins X"],
            "logging": ["Fluentd", "Loki", "Elastic Stack"],
            "ingress": ["NGINX", "Traefik", "HAProxy"],
        },
        aliases=["K8s", "k8s"],
    ))

    graph.add_entity(WebEntity(
        name="Docker Swarm",
        entity_type="technology",
        facts={
            "developed_by": "Docker Inc",
            "release_year": "2015",
        },
        ecosystem={
            "monitoring": ["Prometheus", "cAdvisor"],
        },
    ))

    # Monitoring Tools
    graph.add_entity(WebEntity(
        name="Prometheus",
        entity_type="technology",
        facts={
            "developed_by": "SoundCloud",
            "donated_to": "CNCF",
            "release_year": "2012",
            "written_in": "Go",
        },
        ecosystem={
            "visualization": ["Grafana"],
            "alerting": ["Alertmanager"],
        },
    ))

    graph.add_entity(WebEntity(
        name="Grafana",
        entity_type="technology",
        facts={
            "developed_by": "Grafana Labs",
            "release_year": "2014",
            "headquarters": "New York",
            "founded_year": "2014",
        },
        ecosystem={
            "data_sources": ["Prometheus", "InfluxDB", "Elasticsearch"],
        },
    ))

    graph.add_entity(WebEntity(
        name="Datadog",
        entity_type="technology",
        facts={
            "developed_by": "Datadog Inc",
            "founded_year": "2010",
            "headquarters": "New York",
            "founded_in": "New York",
        },
        ecosystem={
            "integrations": ["AWS", "Kubernetes", "Docker"],
        },
    ))

    graph.add_entity(WebEntity(
        name="New Relic",
        entity_type="technology",
        facts={
            "developed_by": "New Relic Inc",
            "founded_year": "2008",
            "headquarters": "San Francisco",
        },
    ))

    # Cloud Providers
    graph.add_entity(WebEntity(
        name="AWS",
        entity_type="technology",
        facts={
            "developed_by": "Amazon",
            "launch_year": "2006",
            "headquarters": "Seattle",
            "full_name": "Amazon Web Services",
        },
        ecosystem={
            "monitoring": ["CloudWatch", "X-Ray"],
            "database": ["RDS", "DynamoDB", "Aurora", "Redshift"],
            "compute": ["EC2", "Lambda", "ECS", "EKS"],
            "storage": ["S3", "EBS", "EFS"],
        },
        aliases=["Amazon Web Services"],
    ))

    graph.add_entity(WebEntity(
        name="Azure",
        entity_type="technology",
        facts={
            "developed_by": "Microsoft",
            "launch_year": "2010",
            "headquarters": "Redmond",
            "full_name": "Microsoft Azure",
        },
        ecosystem={
            "monitoring": ["Azure Monitor", "Application Insights"],
            "database": ["Cosmos DB", "SQL Database"],
        },
        aliases=["Microsoft Azure"],
    ))

    graph.add_entity(WebEntity(
        name="Google Cloud",
        entity_type="technology",
        facts={
            "developed_by": "Google",
            "launch_year": "2008",
            "headquarters": "Mountain View",
            "full_name": "Google Cloud Platform",
        },
        ecosystem={
            "monitoring": ["Cloud Monitoring", "Cloud Trace"],
            "database": ["BigQuery", "Cloud SQL", "Spanner"],
        },
        aliases=["GCP", "Google Cloud Platform"],
    ))

    # Databases
    graph.add_entity(WebEntity(
        name="PostgreSQL",
        entity_type="technology",
        facts={
            "developed_by": "PostgreSQL Global Development Group",
            "release_year": "1996",
            "origin": "UC Berkeley",
            "written_in": "C",
        },
        aliases=["Postgres"],
    ))

    graph.add_entity(WebEntity(
        name="MongoDB",
        entity_type="technology",
        facts={
            "developed_by": "MongoDB Inc",
            "release_year": "2009",
            "headquarters": "New York",
            "founded_year": "2007",
        },
    ))

    graph.add_entity(WebEntity(
        name="Redis",
        entity_type="technology",
        facts={
            "developed_by": "Salvatore Sanfilippo",
            "release_year": "2009",
            "written_in": "C",
            "company": "Redis Ltd",
            "headquarters": "Mountain View",
        },
    ))

    # Enterprise Software
    graph.add_entity(WebEntity(
        name="Salesforce",
        entity_type="technology",
        facts={
            "developed_by": "Salesforce Inc",
            "founded_year": "1999",
            "headquarters": "San Francisco",
            "founder": "Marc Benioff",
        },
        ecosystem={
            "integrations": ["Slack", "Tableau", "MuleSoft"],
        },
    ))

    graph.add_entity(WebEntity(
        name="SAP",
        entity_type="technology",
        facts={
            "developed_by": "SAP SE",
            "founded_year": "1972",
            "headquarters": "Walldorf, Germany",
            "country": "Germany",
        },
    ))

    graph.add_entity(WebEntity(
        name="Workday",
        entity_type="technology",
        facts={
            "developed_by": "Workday Inc",
            "founded_year": "2005",
            "headquarters": "Pleasanton, California",
            "founders": "Dave Duffield and Aneel Bhusri",
        },
    ))

    # Healthcare IT
    graph.add_entity(WebEntity(
        name="Epic",
        entity_type="technology",
        facts={
            "developed_by": "Epic Systems Corporation",
            "founded_year": "1979",
            "headquarters": "Verona, Wisconsin",
            "founder": "Judith Faulkner",
            "market_share": "largest EHR vendor",
        },
        aliases=["Epic Systems", "Epic EHR"],
    ))

    graph.add_entity(WebEntity(
        name="Cerner",
        entity_type="technology",
        facts={
            "developed_by": "Cerner Corporation",
            "founded_year": "1979",
            "headquarters": "North Kansas City, Missouri",
            "acquired_by": "Oracle",
            "acquisition_year": "2022",
        },
    ))

    # Food Traceability
    graph.add_entity(WebEntity(
        name="FoodLogiQ",
        entity_type="technology",
        facts={
            "developed_by": "FoodLogiQ Inc",
            "founded_year": "2006",
            "headquarters": "Durham, North Carolina",
            "acquired_by": "Trustwell",
        },
    ))

    graph.add_entity(WebEntity(
        name="TraceLink",
        entity_type="technology",
        facts={
            "developed_by": "TraceLink Inc",
            "founded_year": "2009",
            "headquarters": "Wilmington, Massachusetts",
        },
    ))

    # Message Queues
    graph.add_entity(WebEntity(
        name="Kafka",
        entity_type="technology",
        facts={
            "developed_by": "LinkedIn",
            "donated_to": "Apache Software Foundation",
            "release_year": "2011",
            "written_in": "Scala and Java",
        },
        aliases=["Apache Kafka"],
    ))

    graph.add_entity(WebEntity(
        name="RabbitMQ",
        entity_type="technology",
        facts={
            "developed_by": "Pivotal Software",
            "release_year": "2007",
            "written_in": "Erlang",
            "acquired_by": "VMware",
        },
    ))


def _add_company_entities(graph: WebKnowledgeGraph) -> None:
    """Add company entities (developers of technologies)."""

    graph.add_entity(WebEntity(
        name="SoundCloud",
        entity_type="company",
        facts={
            "founded_year": "2007",
            "headquarters": "Berlin",
            "country": "Germany",
            "founders": "Alexander Ljung and Eric Wahlforss",
            "known_for": "audio streaming platform",
        },
    ))

    graph.add_entity(WebEntity(
        name="Google",
        entity_type="company",
        facts={
            "founded_year": "1998",
            "headquarters": "Mountain View, California",
            "founders": "Larry Page and Sergey Brin",
            "parent_company": "Alphabet Inc",
        },
    ))

    graph.add_entity(WebEntity(
        name="Docker Inc",
        entity_type="company",
        facts={
            "founded_year": "2013",
            "headquarters": "Palo Alto, California",
            "founder": "Solomon Hykes",
            "original_name": "dotCloud",
        },
    ))

    graph.add_entity(WebEntity(
        name="Grafana Labs",
        entity_type="company",
        facts={
            "founded_year": "2014",
            "headquarters": "New York",
            "founder": "Torkel Ödegaard",
        },
    ))

    graph.add_entity(WebEntity(
        name="HashiCorp",
        entity_type="company",
        facts={
            "founded_year": "2012",
            "headquarters": "San Francisco",
            "founders": "Mitchell Hashimoto and Armon Dadgar",
            "known_for": "Terraform, Vault, Consul",
        },
    ))


def _add_compliance_entities(graph: WebKnowledgeGraph) -> None:
    """Add compliance/regulation entities."""

    graph.add_entity(WebEntity(
        name="HIPAA",
        entity_type="regulation",
        facts={
            "full_name": "Health Insurance Portability and Accountability Act",
            "enacted_year": "1996",
            "country": "United States",
            "agency": "HHS Office for Civil Rights",
        },
        ecosystem={
            "covered_systems": ["EHR", "patient portals", "PHI storage"],
            "common_vendors": ["Epic", "Cerner", "Meditech"],
        },
        aliases=["Health Insurance Portability and Accountability Act"],
    ))

    graph.add_entity(WebEntity(
        name="FSMA 204",
        entity_type="regulation",
        facts={
            "full_name": "Food Traceability Final Rule",
            "official_title": "Requirements for Additional Traceability Records for Certain Foods",
            "compliance_deadline": "January 2026",
            "agency": "FDA",
            "section": "Section 204(d) of FSMA",
        },
        ecosystem={
            "covered_areas": ["produce traceability", "food safety"],
            "common_vendors": ["FoodLogiQ", "TraceLink"],
        },
        aliases=["Food Traceability Rule", "FDA Food Traceability"],
    ))

    graph.add_entity(WebEntity(
        name="GDPR",
        entity_type="regulation",
        facts={
            "full_name": "General Data Protection Regulation",
            "enacted_year": "2016",
            "effective_year": "2018",
            "jurisdiction": "European Union",
            "max_fine": "4% of annual global turnover",
        },
        aliases=["General Data Protection Regulation"],
    ))

    graph.add_entity(WebEntity(
        name="SOX",
        entity_type="regulation",
        facts={
            "full_name": "Sarbanes-Oxley Act",
            "enacted_year": "2002",
            "country": "United States",
            "primary_focus": "financial reporting and auditing",
        },
        aliases=["Sarbanes-Oxley", "SOX Act"],
    ))

    graph.add_entity(WebEntity(
        name="PCI DSS",
        entity_type="regulation",
        facts={
            "full_name": "Payment Card Industry Data Security Standard",
            "version": "4.0",
            "created_by": "PCI Security Standards Council",
            "founded_year": "2006",
        },
        aliases=["PCI", "Payment Card Industry"],
    ))


def _add_certification_entities(graph: WebKnowledgeGraph) -> None:
    """Add certification and standards entities."""

    # ISO Standards
    graph.add_entity(WebEntity(
        name="ISO 27001",
        entity_type="certification",
        facts={
            "full_name": "Information security management systems",
            "first_published": "2005",
            "organization": "International Organization for Standardization",
            "latest_version": "2022",
        },
        aliases=["ISO/IEC 27001"],
    ))

    graph.add_entity(WebEntity(
        name="ISO 9001",
        entity_type="certification",
        facts={
            "full_name": "Quality management systems",
            "first_published": "1987",
            "organization": "International Organization for Standardization",
            "latest_version": "2015",
        },
    ))

    graph.add_entity(WebEntity(
        name="ISO 14001",
        entity_type="certification",
        facts={
            "full_name": "Environmental management systems",
            "first_published": "1996",
            "organization": "International Organization for Standardization",
            "latest_version": "2015",
        },
    ))

    graph.add_entity(WebEntity(
        name="ISO 22000",
        entity_type="certification",
        facts={
            "full_name": "Food safety management systems",
            "first_published": "2005",
            "organization": "International Organization for Standardization",
            "latest_version": "2018",
        },
    ))

    # SOC Reports
    graph.add_entity(WebEntity(
        name="SOC 2",
        entity_type="certification",
        facts={
            "full_name": "Service Organization Control 2",
            "developed_by": "AICPA",
            "trust_principles": "5",
            "focus": "security, availability, processing integrity, confidentiality, privacy",
        },
        aliases=["SOC 2 Type II", "SOC2"],
    ))

    # Green Building
    graph.add_entity(WebEntity(
        name="LEED",
        entity_type="certification",
        facts={
            "full_name": "Leadership in Energy and Environmental Design",
            "developed_by": "U.S. Green Building Council",
            "founded_year": "1993",
            "headquarters": "Washington, D.C.",
            "platinum_points": "80",
            "gold_points": "60",
            "silver_points": "50",
        },
        aliases=["LEED Certification"],
    ))

    graph.add_entity(WebEntity(
        name="U.S. Green Building Council",
        entity_type="organization",
        facts={
            "abbreviation": "USGBC",
            "founded_year": "1993",
            "headquarters": "Washington, D.C.",
            "known_for": "LEED certification",
        },
        aliases=["USGBC"],
    ))

    # Organic Certification
    graph.add_entity(WebEntity(
        name="Oregon Tilth",
        entity_type="organization",
        facts={
            "full_name": "Oregon Tilth Certified Organic",
            "founded_year": "1974",
            "headquarters": "Corvallis, Oregon",
            "type": "organic certifier",
        },
        aliases=["OTCO", "Oregon Tilth Certified Organic"],
    ))

    graph.add_entity(WebEntity(
        name="CCOF",
        entity_type="organization",
        facts={
            "full_name": "California Certified Organic Farmers",
            "founded_year": "1973",
            "headquarters": "Santa Cruz, California",
            "type": "organic certifier",
        },
        aliases=["California Certified Organic Farmers"],
    ))

    graph.add_entity(WebEntity(
        name="QAI",
        entity_type="organization",
        facts={
            "full_name": "Quality Assurance International",
            "founded_year": "1989",
            "headquarters": "San Diego, California",
            "type": "organic certifier",
        },
        aliases=["Quality Assurance International"],
    ))

    # Maturity Models
    graph.add_entity(WebEntity(
        name="CMMI",
        entity_type="certification",
        facts={
            "full_name": "Capability Maturity Model Integration",
            "developed_by": "Software Engineering Institute",
            "organization": "Carnegie Mellon University",
            "level_3_areas": "18",
            "level_5_areas": "22",
        },
    ))

    # Industry Associations
    graph.add_entity(WebEntity(
        name="CNCF",
        entity_type="organization",
        facts={
            "full_name": "Cloud Native Computing Foundation",
            "founded_year": "2015",
            "headquarters": "San Francisco",
            "parent": "Linux Foundation",
            "known_for": "Kubernetes, Prometheus, Envoy",
        },
        aliases=["Cloud Native Computing Foundation"],
    ))

    graph.add_entity(WebEntity(
        name="AICPA",
        entity_type="organization",
        facts={
            "full_name": "American Institute of Certified Public Accountants",
            "founded_year": "1887",
            "headquarters": "Durham, North Carolina",
            "known_for": "SOC reports, CPA certification",
        },
        aliases=["American Institute of CPAs"],
    ))


# Pre-built graph for easy import
WEB_KNOWLEDGE = build_web_knowledge_graph()


def get_web_knowledge() -> WebKnowledgeGraph:
    """Get the pre-built web knowledge graph."""
    return WEB_KNOWLEDGE

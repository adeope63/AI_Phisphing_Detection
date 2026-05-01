import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


SUSPICIOUS_TLDS = {
    'zip', 'xyz', 'top', 'click', 'work', 'country', 'gq', 'tk', 'ml', 'ga', 'cf'
}

NUMERIC_FEATURE_COLUMNS = [
    'url_count',
    'ip_url_count',
    'https_url_count',
    'suspicious_tld_url_count',
    'sender_domain_matches_any_url_domain',
    'subject_exclamation_count',
    'body_exclamation_count',
    'body_contains_urgent_terms'
]


def _clean_text_fields(dataframe):
    df = dataframe.copy()
    for col in ['sender', 'subject', 'body', 'urls']:
        df[col] = df[col].fillna('').astype(str)
    return df


def combine_text_columns(dataframe):
    """
    Combine sender, subject, body, and urls into one text field.
    """
    df = _clean_text_fields(dataframe)
    return (
        df['sender'] + ' ' +
        df['subject'] + ' ' +
        df['body'] + ' ' +
        df['urls']
    )


def _extract_urls(urls_text):
    if not urls_text:
        return []
    return re.findall(r'(https?://[^\s,;]+|www\.[^\s,;]+)', urls_text.lower())


def _sender_domain(sender):
    sender = (sender or '').lower().strip()
    if '@' not in sender:
        return ''
    return sender.split('@')[-1]


def _url_domain(url_value):
    normalized = url_value if url_value.startswith(('http://', 'https://')) else f'http://{url_value}'
    domain = urlparse(normalized).netloc.lower()
    return domain.split(':')[0]


def _is_ip_host(domain):
    return bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', domain))


def _has_urgent_terms(text):
    text = (text or '').lower()
    urgent_terms = ['urgent', 'verify', 'suspend', 'password', 'immediately', 'limited time', 'confirm']
    return int(any(term in text for term in urgent_terms))


def build_feature_frame(dataframe):
    """
    Build combined text + engineered numeric phishing features.
    """
    df = _clean_text_fields(dataframe)

    url_lists = df['urls'].apply(_extract_urls)
    url_domains = url_lists.apply(lambda urls: [_url_domain(url) for url in urls if _url_domain(url)])
    sender_domains = df['sender'].apply(_sender_domain)

    feature_df = pd.DataFrame(index=df.index)
    feature_df['combined_text'] = combine_text_columns(df)
    feature_df['url_count'] = url_lists.apply(len).astype(float)
    feature_df['ip_url_count'] = url_domains.apply(lambda domains: sum(_is_ip_host(d) for d in domains)).astype(float)
    feature_df['https_url_count'] = url_lists.apply(
        lambda urls: sum(u.startswith('https://') for u in urls)
    ).astype(float)
    feature_df['suspicious_tld_url_count'] = url_domains.apply(
        lambda domains: sum((d.split('.')[-1] if '.' in d else '') in SUSPICIOUS_TLDS for d in domains)
    ).astype(float)
    feature_df['sender_domain_matches_any_url_domain'] = [
        float(bool(sender_domain and any(sender_domain in domain or domain in sender_domain for domain in domains)))
        for sender_domain, domains in zip(sender_domains, url_domains)
    ]
    feature_df['subject_exclamation_count'] = df['subject'].str.count('!').astype(float)
    feature_df['body_exclamation_count'] = df['body'].str.count('!').astype(float)
    feature_df['body_contains_urgent_terms'] = df['body'].apply(_has_urgent_terms).astype(float)

    return feature_df


def build_preprocessor():
    """
    Create preprocessing pipeline that mixes TF-IDF text and engineered numeric features.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            (
                'tfidf_text',
                TfidfVectorizer(
                    lowercase=True,
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=5000
                ),
                'combined_text'
            ),
            ('numeric_features', 'passthrough', NUMERIC_FEATURE_COLUMNS)
        ]
    )

    return Pipeline([
        ('feature_builder', FunctionTransformer(build_feature_frame, validate=False)),
        ('feature_union', column_transformer)
    ])
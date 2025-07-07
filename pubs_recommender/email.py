import smtplib
from email.mime.text import MIMEText
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import os
from datetime import datetime

def send_email_with_SMTP(message, to_address):
    """
    Send email using SMTP (SendGrid).
    
    Args:
        message: Email message content
        to_address: Recipient email address
    """
    # Email account details
    SMTP_SERVER = "smtp.sendgrid.net"
    SMTP_PORT = 587
    SENDGRID_API_KEY = os.getenv('SG_API_KEY')  # Replace with your actual API key
    EMAIL_ADDRESS = os.getenv('SENDER_EMAIL')

    # Recipient details
    TO_EMAIL = to_address
    SUBJECT = "Publications of Interest"
    BODY = message

    # Create the email
    msg = MIMEText(BODY)
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    # Send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login("apikey", SENDGRID_API_KEY)  # Use "apikey" as the username
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
            server.quit()
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
        
def send_email_with_Web_API(message, to_address, html_content=None):
    """
    Send email using SendGrid Web API.
    
    Args:
        message: Plain text email message
        to_address: Recipient email address
        html_content: HTML version of the email (optional)
    """
    # Email configuration
    SENDGRID_API_KEY = os.getenv('SG_API_KEY')
    FROM_EMAIL = "bot@phillygenome.xyz"
    TO_EMAIL = to_address
    SUBJECT = "Publications of Interest"
    BODY = message

    # Create the email message
    if html_content:
        msg = Mail(
            from_email=FROM_EMAIL,
            to_emails=TO_EMAIL,
            subject=SUBJECT,
            plain_text_content=BODY,
            html_content=html_content
        )
    else:
        msg = Mail(
            from_email=FROM_EMAIL,
            to_emails=TO_EMAIL,
            subject=SUBJECT,
            plain_text_content=BODY
        )

    try:
        # Create SendGrid client
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        # Send the email
        response = sg.send(msg)
        print(f"Email sent successfully! Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def draft_plaintext_email(pubs, papers, bow_corpus, similarity_matrix, dictionary, topic_results):
    """
    Draft a plain text email with publication recommendations.
    Args:
        pubs: List of publications
        papers: List of papers from library
        bow_corpus: Bag-of-words corpus
        similarity_matrix: Similarity matrix
        dictionary: Gensim dictionary
        topic_results: Dictionary mapping cluster IDs to matching publications
    Returns:
        Plain text email content
    """
    from .language import extract_lda_keywords, extract_cluster_docs
    # Get all clusters present in the library
    unique_clusters = sorted(set(p.cluster_id for p in papers if getattr(p, 'cluster_id', None) is not None))

    if not pubs:
        return "No new publications this week."

    message = "Your Literature Digest for This Week\n"
    message += "=" * 40 + "\n\n"

    section_matches = "Here is a list of recent publications that match your interest.\n"
    section_matches += "-" * 40 + "\n\n"
    section_no_matches = "Here are your interests without a match this week.\n"
    section_no_matches += "-" * 40 + "\n\n"

    for cluster in unique_clusters:
        extracted_corpus = extract_cluster_docs(bow_corpus, papers, cluster_num=cluster)
        keywords = extract_lda_keywords(dictionary, extracted_corpus)
        keywords_str = ", ".join([k for k, _ in keywords])
        matching_pubs = topic_results.get(cluster, [])

        if len(matching_pubs) != 0:
            topic_desc = f"Because you've been reading about a topic with these keywords: {keywords_str}\n"
            section_matches += topic_desc
            for pub in matching_pubs:
                title = getattr(pub, 'title', 'No title')
                authors = getattr(pub, 'authors', '')
                if isinstance(authors, list):
                    author_names = []
                    for a in authors:
                        if isinstance(a, dict) and 'name' in a:
                            author_names.append(a['name'])
                        elif hasattr(a, 'name'):
                            author_names.append(a.name)
                        else:
                            author_names.append(str(a))
                    authors = ', '.join(author_names)
                authors = authors if authors else ''
                journal = getattr(pub, 'prism_publicationname', getattr(pub, 'journal', ''))
                if hasattr(pub, 'prism_doi') and pub.prism_doi:
                    link = f"https://doi.org/{pub.prism_doi}"
                else:
                    link = getattr(pub, 'link', '')
                section_matches += f"  - {title}\n"
                if authors:
                    section_matches += f"    Authors: {authors}\n"
                if journal:
                    section_matches += f"    Journal: {journal}\n"
                if link:
                    section_matches += f"    Read more: {link}\n"
                section_matches += "\n"
            section_matches += "-" * 20 + "\n\n"
        else:
            topic_desc = f"Your topic of interest based on keywords: {keywords_str}\n"
            section_no_matches += topic_desc + "\n"

    message += section_matches + section_no_matches
    message += f"\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return message

def draft_html_email(pubs, papers, bow_corpus, similarity_matrix, dictionary, topic_results):
    """
    Draft an HTML email with publication recommendations.
    Args:
        pubs: List of publications to recommend
        papers: List of Paper objects with cluster IDs
        bow_corpus: Bag of words corpus for new publications
        similarity_matrix: Similarity matrix between papers and publications
        dictionary: Gensim dictionary
        topic_results: Dictionary mapping cluster IDs to lists of matching publications
    """
    from .language import extract_lda_keywords, extract_cluster_docs
    # unique_clusters: all clusters present in the library, not just those with matches
    unique_clusters = sorted(set(p.cluster_id for p in papers if getattr(p, 'cluster_id', None) is not None))
    
    # If no new papers, return early
    if not pubs:
        return "<html><body><p>No new publications this week.</p></body></html>"

    # Generate the HTML email content (light theme, blue accents, card containers)
    html_message = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; background: #f8f9fa; color: #222; }
            h1 { color: #2c3e50; border-bottom: 10px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #fff; border-radius: 10px; box-shadow: 0 1px 6px #0001; }
            .paper-item { margin: 14px 0; padding: 14px; background-color: #f4f8fb; border-radius: 7px; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
            .paper-title { font-weight: bold; color: #2c3e50; font-size: 1.1em; margin-bottom: 2px; }
            .paper-authors { color: #1565c0; font-size: 0.98em; margin-bottom: 2px; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .paper-journal { color: #7f8c8d; font-style: italic; margin-bottom: 2px; }
            .paper-link { color: #3498db; text-decoration: underline; }
            .paper-link:hover { text-decoration: underline; }
            .keywords { color: #36454f; font-style: italic; }
            .keyword-token { display: inline-block; background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em; font-weight: 500; }
            .divider { border-top: 4px solid #bdc3c7; margin: 20px 0; }
            .section-header { font-weight: bold; margin-bottom: 14px; font-size: 1.2em; color: #222; }
        </style>
    </head>
    <body>
    """
    
    html_message += "<h1>Your Literature Digest for This Week</h1>"

    section_matches = "<div class='section'><div class='section-header'>Here is a list of recent publications that match your interest.</div>"
    section_no_matches = "<div class='section'><div class='section-header'>Here are your interests without a match this week.</div>"

    for cluster in unique_clusters:
        extracted_corpus = extract_cluster_docs(bow_corpus, papers, cluster_num=cluster)
        topic_keys = extract_lda_keywords(dictionary, extracted_corpus)
        matching_pubs = topic_results.get(cluster, [])  # Use .get to allow for unmatched topics
        
        if len(matching_pubs) != 0:
            # Format all topic keywords
            topic_desc = "<p class='keywords'><strong>Because you've been reading about a topic with these keywords:</strong> "
            for keyword, score in topic_keys:
                topic_desc += f"<span class='keyword-token'>{keyword}</span> "
            topic_desc += "</p>"
            section_matches += topic_desc
            
            for pub in matching_pubs:
                if hasattr(pub, 'prism_doi') and pub.prism_doi:
                    link = f"https://doi.org/{pub.prism_doi}"
                else:
                    link = getattr(pub, 'link', '')
                title = getattr(pub, 'title', 'No title')
                authors = getattr(pub, 'authors', '')
                if isinstance(authors, list):
                    author_names = []
                    for a in authors:
                        if isinstance(a, dict) and 'name' in a:
                            author_names.append(a['name'])
                        elif hasattr(a, 'name'):
                            author_names.append(a.name)
                        else:
                            author_names.append(str(a))
                    authors = ', '.join(author_names)
                authors = authors if authors else ''
                journal = getattr(pub, 'prism_publicationname', getattr(pub, 'journal', ''))
                section_matches += f"""
                <div class='paper-item'>
                    <div class='paper-title'>{title}</div>
                    <div class='paper-authors'>{authors}</div>
                    <div class='paper-journal'>{journal}</div>
                    <div><a href='{link}' class='paper-link'>Read more</a></div>
                </div>
                """
            section_matches += "<div class='divider'></div>"
        else:
            # Format all keywords for topics without matches
            topic_desc = "<p class='keywords'><strong>Your topic of interest based on keywords:</strong> "
            for keyword, score in topic_keys:
                topic_desc += f"<span class='keyword-token'>{keyword}</span> "
            topic_desc += "</p>"
            section_no_matches += topic_desc
        
    html_message += section_matches + "</div>" + section_no_matches + "</div>"
    html_message += "</body></html>"
    
    return html_message 
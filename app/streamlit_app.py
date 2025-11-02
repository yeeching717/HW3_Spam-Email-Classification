from pathlib import Path
import streamlit as st


def load_proposal_md() -> str:
    # proposal is under openspec/changes/add-spam-classification-baseline/proposal.md
    repo_root = Path(__file__).resolve().parent.parent
    proposal_path = (
        repo_root
        / "openspec"
        / "changes"
        / "add-spam-classification-baseline"
        / "proposal.md"
    )
    if not proposal_path.exists():
        return "# Proposal file not found\n\nExpected: {}".format(proposal_path)
    return proposal_path.read_text(encoding="utf-8")


def main():
    st.set_page_config(page_title="Spam classification proposal", layout="wide")
    st.title("Add: Spam Classification Baseline — Proposal")

    st.sidebar.header("Proposal viewer")
    st.sidebar.markdown("Displays the change proposal from the `openspec` folder.")

    md = load_proposal_md()

    # render markdown
    st.markdown(md, unsafe_allow_html=False)


if __name__ == "__main__":
    main()

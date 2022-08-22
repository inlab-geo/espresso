from pathlib import Path
from shutil import copy
import os
import yaml

import cofi_espresso as esp


def gen_contrib_docs(_):
    all_contribs = [nm for nm in dir(esp) if not nm.startswith("_") and nm[0].islower()]
    base_path = esp.__path__[0]
    dest_path = Path(__file__).resolve().parent.parent / "user_guide" / "contrib" / "generated"
    os.mkdir(dest_path)
    for contrib in all_contribs:
        contrib_dir = Path(f"{base_path}/{contrib}")
        dest_contrib_dir = Path(f"{dest_path}/{contrib}")
        if contrib_dir.exists() and contrib_dir.is_dir():
            # -> locate files
            file_metadata = contrib_dir / "metadata.yml"
            file_readme = contrib_dir / "README.md"
            file_licence = contrib_dir / "LICENCE"
            # -> make new folder docs/source/contrib/<contrib-name>
            os.mkdir(dest_contrib_dir)
            # -> copy README and LICENCE
            copy(file_readme, f"{dest_contrib_dir}/README.md")
            copy(file_licence, f"{dest_contrib_dir}/LICENCE")
            with open(file_metadata, "r") as f:
                metadata = yaml.safe_load(f)
            lines = []
            # -> include README.md
            lines.append("```{include} ./README.md\n```")
            # -> format metadata files
            lines.append(":::{admonition} Contribution Metadata for ")
            lines[-1] += f"{metadata['name']} \n:class: important"
            # metadata - short description
            lines.append(metadata['short_description'])
            lines.append("```{eval-rst}")
            # metadata - authors
            lines.append("\n:Author: " + ", ".join(metadata["authors"]))
            # metadata - contacts
            lines.append(":Contact:")
            if "contacts" in metadata and len(metadata["contacts"]) > 0:
                if len(metadata["contacts"]) == 1:
                    contact = metadata["contacts"][0]
                    lines.append(f"  {contact['name']} ({contact['email']}) ")
                else:
                    for contact in metadata["contacts"]:
                        lines.append(f"  - {contact['name']} {contact['email']} ")
            # metadata - citations
            if "citations" in metadata and len(metadata["citations"]) > 0:
                lines.append(":Citation:")
                if len(metadata["citations"]) == 1:
                    citation = metadata["citations"][0]
                    lines.append(f"  doi: {citation['doi']}")
                else:
                    for citation in metadata["citations"]:
                        lines.append(f"  - doi: {citation['doi']}")
            # metadata - extra website
            if "extra_websites" in metadata and len(metadata["extra_websites"]) > 0:
                lines.append(":Extra website:")
                if len(metadata["extra_websites"]) == 1:
                    extra_website = metadata["extra_websites"][0]
                    lines.append(f"  [{extra_website['name']}]({extra_website['link']})")
                else:
                    for extra_website in metadata["extra_websites"]:
                        lines.append(f"  - `{extra_website['name']} <{extra_website['link']}>`_")
            # metadata - examples
            lines.append(":Examples:\n")
            lines.append("  .. list-table::")
            lines.append("    :widths: 10 40 25 25")
            lines.append("    :header-rows: 1")
            lines.append("    :stub-columns: 1\n")
            lines.append("    * - index")
            lines.append("      - description")
            lines.append("      - model dimension")
            lines.append("      - data dimension")
            for idx, example in enumerate(metadata["examples"]):
                lines.append(f"    * - {idx+1}")
                lines.append(f"      - {example['description']}")
                lines.append(f"      - {example['model_dimension']}")
                lines.append(f"      - {example['data_dimension']}")
            lines.append("```")
            lines.append(":::")
            # -> include LICENCE
            lines.append("## LICENCE\n")
            lines.append("```{include} ./LICENCE\n```")
            # -> write to index.md file
            with open(f"{dest_contrib_dir}/index.md", "w") as f:
                f.write("\n".join(lines))
            # -> add contrib link to contrib/index.rst
            with open(Path(dest_path).parent / "_index.rst", "r") as f:
                index_template = f.read()
            with open(Path(dest_path).parent / "index.rst", "w") as f:
                f.write(index_template)
                f.write(f"    generated/{contrib}/index.md")

def setup(app):
    app.connect("builder-inited", gen_contrib_docs)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

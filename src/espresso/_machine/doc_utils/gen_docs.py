r"""
Module to contain funtions that help generate documentation
"""
from pathlib import Path
from shutil import copy
import os
import espresso as esp


def read_metadata(contrib_name, lines):
    # - problem_title
    # - problem_short_description
    # - author_names
    # - contact_name
    # - contact_email
    # - [optional] citations -> []
    # - [optional] linked_sites -> [(name, link)]
    contrib_class_name = contrib_name.title().replace("_", "")
    contrib_class = getattr(esp, contrib_class_name)
    class_metadata = contrib_class.metadata
    lines.append(":::{admonition} Contribution Metadata for ")
    lines[-1] += "*" + class_metadata["problem_title"] + "* \n:class: important"
    # metadata - short description
    lines.append(class_metadata["problem_short_description"])
    lines.append("```{eval-rst}")
    # metadata - authors
    lines.append("\n:Author: " + ", ".join(class_metadata["author_names"]))
    # metadata - contact
    lines.append(
        ":Contact: "
        + class_metadata["contact_name"]
        + " ("
        + class_metadata["contact_email"]
        + ")"
    )
    # metadata - citations
    if len(class_metadata["citations"]) > 0:
        lines.append(":Citation:")
        if len(class_metadata["citations"]) == 1:
            citation = class_metadata["citations"][0]
            lines.append(f"  {citation[0]}")
            if citation[1]:
                lines[-1] += f", {citation[1]}"
        else:
            for citation in class_metadata["citations"]:
                lines.append(f"  - {citation[0]}")
                if citation[1]:
                    lines[-1] += f", {citation[1]}"
    # metadata - extra website
    if len(class_metadata["linked_sites"]) > 0:
        lines.append(":Extra website:")
        if len(class_metadata["linked_sites"]) == 1:
            name, link = class_metadata["linked_sites"][0]
            lines.append(f"  `{name} <{link}>`_")
        else:
            for (name, link) in class_metadata["linked_sites"]:
                lines.append(f"  - [{name}]({link})")
    # ending
    lines.append("```")
    lines.append(":::")


def read_file(contrib_dir, dest_contrib_dir, file_name, lines):
    src_path = contrib_dir / file_name
    dst_path = dest_contrib_dir / file_name
    copy(src_path, dst_path)
    lines.append("```{include} ./" + file_name + "\n```")


def contribs(BASE_PATH, DEST_PATH):
    names = [cls.__module__.split(".")[-1] for cls in esp.list_problems()]
    all_contribs = [
        (contrib, Path(f"{BASE_PATH}/{contrib}"), Path(f"{DEST_PATH}/{contrib}"))
        for contrib in names
    ]
    all_contribs = [
        contrib
        for contrib in all_contribs
        if contrib[1].exists() and contrib[1].is_dir()
    ]
    return all_contribs


def gen_contrib_docs(BASE_PATH, DEST_PATH):
    os.mkdir(DEST_PATH)
    # prepare index file
    index_lines = []
    # move info for each contribution
    for (contrib, src_folder, dst_folder) in contribs(BASE_PATH, DEST_PATH):
        # make new folder docs/source/contrib/<contrib-name>
        os.mkdir(dst_folder)
        lines = []
        # Provide hint about where this file came from
        lines.append(
            "<!--- This file was automatically generated by \n"
            "      _ext/generate_contrib_docs.py based on the\n"
            "      contributor-supplied metadata and the files\n"
            "      README.md and LICENCE. \n"
            "-->\n\n"
        )
        # include README.md
        read_file(src_folder, dst_folder, "README.md", lines)
        # format metadata content
        read_metadata(contrib, lines)
        # include LICENCE
        lines.append("\n## LICENCE\n")
        read_file(src_folder, dst_folder, "LICENCE", lines)
        # write to index.md file
        with open(f"{dst_folder}/index.md", "w") as f:
            f.write("\n".join(lines))
        # add contrib link to contrib/index.rst
        index_lines.append(f"    generated/{contrib}/index.md\n")
    # write index file
    with open(Path(DEST_PATH).parent / "_index.rst", "r") as f:
        index_template = f.read()
    with open(Path(DEST_PATH).parent / "index.rst", "w") as f:
        f.write(
            "..\n   *** N.B. This file is automatically generated by\n"
            "       _ext/generate_contrib_docs.py based on the template\n"
            "       in _index.rst ***\n\n"
        )
        f.write(index_template)
        for line in index_lines:
            f.write(line)

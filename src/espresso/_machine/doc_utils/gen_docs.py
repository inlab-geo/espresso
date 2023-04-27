r"""
Module to contain functions that help generate documentation
"""
from pathlib import Path
from shutil import copy
import os
import espresso


def read_metadata(contrib_name, lines):
    # - problem_title
    # - problem_short_description
    # - author_names
    # - contact_name
    # - contact_email
    # - [optional] citations -> []
    # - [optional] linked_sites -> [(name, link)]
    contrib_class_name = contrib_name.title().replace("_", "")
    contrib_class = getattr(espresso, contrib_class_name)
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


def write_sample_code(class_name, name, lines):
    capabilities = espresso.list_capabilities(class_name)[class_name]
    problem_var_name = f"my{name}"
    lines.append(f"## Example usage for `{class_name}` \n")
    with open(Path(__file__).resolve().parent / "_sample_code.txt", "r") as f:
        optional = False
        for line in f:
            if "Optional API" in line:
                optional = True
            if "<problem_var_name>." in line:
                attr_name = line.split("<problem_var_name>.")[1].split("(")[0].strip()
                if attr_name not in capabilities and optional:
                    continue
                elif attr_name in capabilities:
                    capabilities.remove(attr_name)
            line = line.replace("<problem_var_name>", problem_var_name)
            line = line.replace("<class_name>", class_name)
            lines.append(line.strip())
    if capabilities:
        additional_api = f"Additional attributes to explore: {capabilities}."
        additional_api = additional_api.replace("'", "`")
        lines.append(additional_api)


def write_example_files(contrib_dir, dest_contrib_dir, lines):
    if "examples" in os.listdir(contrib_dir):
        lines.append("## Example files \n")
        for file_name in os.listdir(contrib_dir / "examples"):
            if file_name.endswith(".pyc") or file_name == "__pycache__":
                continue
            link_file(contrib_dir / "examples", dest_contrib_dir, file_name, lines)
    

def read_file(contrib_dir, dest_contrib_dir, file_name, lines):
    src_path = contrib_dir / file_name
    dst_path = dest_contrib_dir / file_name
    copy(src_path, dst_path)
    lines.append("```{include} ./" + file_name + "\n```\n")


def link_file(contrib_dir, dest_contrib_dir, file_name, lines):
    src_path = contrib_dir / file_name
    dst_path = dest_contrib_dir / "examples" / file_name
    os.makedirs(dest_contrib_dir / "examples", exist_ok=True)
    copy(src_path, dst_path)
    lines.append("- {download}" + f"`examples/{file_name}`")


def contribs(BASE_PATH, DEST_PATH):
    names = [cls.__module__.split(".")[-1] for cls in espresso.list_problems()]
    class_names = espresso.list_problem_names()
    all_contribs = [
        (contrib, Path(f"{BASE_PATH}/{contrib}"), Path(f"{DEST_PATH}/{contrib}"))
        for contrib in names
    ]
    all_contribs = [
        contrib
        for contrib in all_contribs
        if contrib[1].exists() and contrib[1].is_dir()
    ]
    return zip(all_contribs, class_names)


def gen_contrib_docs(BASE_PATH, DEST_PATH):
    os.mkdir(DEST_PATH)
    # prepare index file
    index_lines = []
    # move info for each contribution
    for (contrib, src_folder, dst_folder), class_name in \
        contribs(BASE_PATH, DEST_PATH):
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
        # format sample code based on capability matrix
        write_sample_code(class_name, contrib, lines)
        # format example files (if any)
        write_example_files(src_folder, dst_folder, lines)
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

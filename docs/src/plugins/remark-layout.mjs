export default () => {
  return (tree, file) => {
    file.data.astro.frontmatter.layout = "../../layouts/Layout.astro";
    for (const node of tree.children) {
      if (node.type === "paragraph" && node.children?.length > 1) {
        node.children.push({ type: "mdxJsxFlowElement", name: "span" });
      }
    }
  };
};

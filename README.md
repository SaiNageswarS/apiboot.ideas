# αpiβoot Ideas

A content platform to share knowledge, ideas, and experiences across technology, fitness, and lifestyle.

🌐 **Live Site**: [ideas.apiboot.com](https://ideas.apiboot.com)

## About

αpiβoot Ideas is a curated space to share ideas, experiences, and tools — across tech, fitness, and lifestyle. A community of builders exploring how we learn, create, and live better.

## Contributing Articles

We welcome contributions from the community! To contribute an article:

### 1. Create Author Folder
Create a new folder in the `content/` directory with your GitHub username or preferred author name:
```
content/your-username/
```

### 2. Add Author Image
Add your profile image as `author.jpg` in the `static/images/` directory:
```
static/images/your-username.jpg
```

### 3. Create Your Article
Add your article as a Markdown file in your author folder:
```
content/your-username/your-article-title.md
```

### 4. Article Structure
Your article should include front matter with metadata:

```yaml
---
title: "Your Article Title"
date: 2024-01-01
draft: false
tags: ["tag1", "tag2"]
categories: ["category"]
author: "Your Name"
description: "Brief description of your article"
---
```

### 5. Submit a Pull Request
1. Fork this repository
2. Create your author folder and add your content
3. Submit a pull request with a clear description of your contribution

## Technical Details

- **Framework**: Hugo static site generator
- **Theme**: PaperMod
- **Deployment**: Hosted on ideas.apiboot.com
- **Comments**: Giscus integration for community discussions

## Local Development

To run the site locally:

```bash
# Install Hugo (if not already installed)
# Visit: https://gohugo.io/installation/

# Clone the repository
git clone https://github.com/SaiNageswarS/apiboot.ideas.git
cd apiboot.ideas

# Start the development server
hugo server -D

# The site will be available at http://localhost:1313
```

## Content Guidelines

- Write clear, engaging content that adds value to the community
- Use proper Markdown formatting
- Include relevant tags and categories
- Add a compelling description for better SEO
- Ensure your content is original and properly attributed

## License

Content is shared under appropriate licenses. Please respect intellectual property and provide proper attribution when necessary.

---

**Ready to share your knowledge?** Start by creating your author folder and submitting your first article!

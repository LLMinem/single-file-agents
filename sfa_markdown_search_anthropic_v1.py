#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "anthropic>=0.47.1",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0",
# ]
# ///

"""
Usage:
    # Basic search with default settings
    uv run sfa_markdown_search_anthropic_v1.py -d ~/Documents -p "Find notes about project ideas"
    
    # Search multiple directories with custom settings
    uv run sfa_markdown_search_anthropic_v1.py -d ~/Documents ~/Projects -p "Find documentation about API authentication" --depth 3 --formats md,txt,rst
    
    # Search with exclusions and JSON output
    uv run sfa_markdown_search_anthropic_v1.py -d ~/Obsidian/vault -p "Find notes containing task lists" --exclude "*archive*,*templates*" --output results.json
"""

import os
import sys
import json
import time
import fnmatch
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import mimetypes
from dataclasses import dataclass, asdict, field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table
from rich import box
import anthropic

# Constants
MAX_PREVIEW_LINES = 10
DEFAULT_DEPTH = 5
DEFAULT_FORMATS = ["md", "txt"]
DEFAULT_RESULT_LIMIT = 20
DEFAULT_COMPUTE_LIMIT = 30
PREVIEW_BATCH_SIZE = 5
CONTENT_BATCH_SIZE = 2
RELEVANCE_THRESHOLD = 25  # Minimum score (0-100) to consider a file relevant

# Initialize rich console
console = Console()


@dataclass
class FileResult:
    """Data class for file search results."""
    file_path: str
    score: int = 0  # 0-100 relevance score
    reason: str = ""
    preview_text: str = ""
    is_relevant: bool = False
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return asdict(self)


class FileDiscovery:
    """Handles finding all matching files in target directories."""
    
    def __init__(
        self, 
        directories: List[str], 
        formats: List[str] = DEFAULT_FORMATS,
        max_depth: int = DEFAULT_DEPTH,
        exclude_patterns: List[str] = None
    ):
        self.directories = [os.path.abspath(d) for d in directories]
        self.formats = formats
        self.max_depth = max_depth
        self.exclude_patterns = exclude_patterns or []
        # Initialize mimetypes
        if not mimetypes.inited:
            mimetypes.init()
    
    def _is_text_file(self, file_path: str) -> bool:
        """Check if a file is a text file based on content and mimetype."""
        # First check by extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext and ext[1:] in self.formats:
            mime_type = mimetypes.guess_type(file_path)[0]
            # If we can determine the mime type and it's text, accept it
            if mime_type and mime_type.startswith('text/'):
                return True
                
            # For unknown mime types, try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)  # Try reading a bit of the file
                return True
            except UnicodeDecodeError:
                return False
        return False
    
    def _should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded based on patterns."""
        path = os.path.abspath(path)
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False
    
    def find_files(self, progress_callback=None) -> List[str]:
        """Find all matching files in the specified directories."""
        matching_files = []
        
        for directory in self.directories:
            if not os.path.exists(directory):
                console.log(f"[yellow]Warning: Directory '{directory}' does not exist[/yellow]")
                continue
                
            # Walk the directory tree
            root_depth = directory.count(os.path.sep)
            for root, dirs, files in os.walk(directory):
                # Check depth
                current_depth = root.count(os.path.sep) - root_depth
                if current_depth > self.max_depth:
                    dirs.clear()  # Don't go deeper
                    continue
                    
                # Remove excluded directories (modify dirs in-place)
                dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
                
                # Find matching files
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip excluded files
                    if self._should_exclude(file_path):
                        continue
                        
                    # Check if it's a text file with matching extension
                    if self._is_text_file(file_path):
                        matching_files.append(file_path)
                        if progress_callback:
                            progress_callback(file_path)
        
        return matching_files


class FileAnalyzer:
    """Handles analyzing file content for relevance."""
    
    def __init__(self, client: anthropic.Anthropic, search_prompt: str):
        self.client = client
        self.search_prompt = search_prompt
        self.preview_model = "claude-3-5-haiku-20240307"  # Faster model for preview analysis
        self.content_model = "claude-3-7-sonnet-20250219"  # More powerful model for content analysis
        
    def read_file_preview(self, file_path: str, preview_lines: int = MAX_PREVIEW_LINES) -> str:
        """Read a preview of the file (first N lines)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ''.join(f.readline() for _ in range(preview_lines))
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def read_full_content(self, file_path: str, max_size: int = 100000) -> str:
        """Read the full content of a file, truncating if too large."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(max_size)
                if len(content) == max_size:
                    content += "\n[... File truncated due to size ...]"
                return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def analyze_preview_batch(self, file_paths: List[str], preview_lines: int = MAX_PREVIEW_LINES) -> List[FileResult]:
        """Analyze a batch of file previews for potential relevance."""
        results = []
        file_previews = {}
        
        # First read all previews
        for file_path in file_paths:
            preview = self.read_file_preview(file_path, preview_lines)
            file_previews[file_path] = preview
        
        # Construct the batch prompt
        batch_prompt = f"""<purpose>
You are helping evaluate if file previews match a search prompt.
</purpose>

<instructions>
- Evaluate if each file PREVIEW might contain content relevant to the search prompt
- Judge ONLY based on the preview text provided
- Assign a preliminary relevance score (0-100) for each file
- Score 0 means definitely not relevant
- Score 100 means definitely relevant
- Provide a very brief reason for each score
- Be strict - only high scores if preview clearly matches search prompt
- Preview analysis is preliminary screening only
</instructions>

<search_prompt>
{self.search_prompt}
</search_prompt>

"""
        # Add each file to the batch prompt
        for i, (file_path, preview) in enumerate(file_previews.items()):
            file_name = os.path.basename(file_path)
            batch_prompt += f"""
<file_{i+1}>
File: {file_name}
Preview:
{preview}
</file_{i+1}>
"""
        
        batch_prompt += """
Format your response as JSON with this structure for each file:
```json
[
  {
    "file_id": "file_1",
    "score": 75,  
    "reason": "Brief reason for score",
    "should_analyze_full": true/false
  },
  ...
]
```
IMPORTANT: Return ONLY valid JSON, no other text. 'should_analyze_full' should be true for scores above 25.
"""
        
        try:
            response = self.client.messages.create(
                model=self.preview_model,
                max_tokens=1000,
                system="You analyze file previews to determine if they might be relevant to a search query. You MUST respond with valid JSON only, no other text.",
                messages=[
                    {"role": "user", "content": batch_prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from the response
            try:
                # Try to parse the response as JSON directly
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract it from code blocks
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                else:
                    # Last resort - try to extract anything that looks like JSON
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract JSON from response")
            
            # Map the analysis to file results
            for i, result in enumerate(analysis):
                file_id = result.get("file_id", f"file_{i+1}")
                file_index = int(file_id.split("_")[1]) - 1
                
                if file_index < len(file_paths):
                    file_path = file_paths[file_index]
                    score = result.get("score", 0)
                    reason = result.get("reason", "")
                    should_analyze_full = result.get("should_analyze_full", score > RELEVANCE_THRESHOLD)
                    
                    results.append(FileResult(
                        file_path=file_path,
                        score=score,
                        reason=reason,
                        preview_text=file_previews[file_path],
                        is_relevant=should_analyze_full
                    ))
        except Exception as e:
            # If batch analysis fails, fall back to individual analysis
            console.log(f"[yellow]Batch analysis failed: {str(e)}[/yellow]")
            console.log("[yellow]Falling back to individual analysis[/yellow]")
            
            for file_path in file_paths:
                preview = file_previews.get(file_path, self.read_file_preview(file_path, preview_lines))
                try:
                    result = self._analyze_preview_individual(file_path, preview)
                    results.append(result)
                except Exception as e:
                    results.append(FileResult(
                        file_path=file_path,
                        error=f"Error analyzing preview: {str(e)}"
                    ))
            
        return results
    
    def _analyze_preview_individual(self, file_path: str, preview: str) -> FileResult:
        """Analyze a single file preview for potential relevance."""
        file_name = os.path.basename(file_path)
        
        prompt = f"""<purpose>
You are helping evaluate if a file preview matches a search prompt.
</purpose>

<instructions>
- Evaluate if the file PREVIEW might contain content relevant to the search prompt
- Judge ONLY based on the preview text provided
- Assign a preliminary relevance score (0-100)
- Score 0 means definitely not relevant
- Score 100 means definitely relevant
- Provide a very brief reason for your score
- Be strict - only high scores if preview clearly matches search prompt
</instructions>

<search_prompt>
{self.search_prompt}
</search_prompt>

<file>
File: {file_name}
Preview:
{preview}
</file>

Format your response as JSON with this structure:
```json
{
  "score": 75,  
  "reason": "Brief reason for score",
  "should_analyze_full": true/false
}
```
IMPORTANT: Return ONLY valid JSON, no other text. 'should_analyze_full' should be true for scores above 25.
"""
        
        try:
            response = self.client.messages.create(
                model=self.preview_model,
                max_tokens=300,
                system="You analyze file previews to determine if they might be relevant to a search query. You MUST respond with valid JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from the response
            try:
                # Try to parse the response as JSON directly
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract it from code blocks
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                else:
                    # Last resort - try to extract anything that looks like JSON
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract JSON from response")
            
            score = analysis.get("score", 0)
            reason = analysis.get("reason", "")
            should_analyze_full = analysis.get("should_analyze_full", score > RELEVANCE_THRESHOLD)
            
            return FileResult(
                file_path=file_path,
                score=score,
                reason=reason,
                preview_text=preview,
                is_relevant=should_analyze_full
            )
            
        except Exception as e:
            return FileResult(
                file_path=file_path,
                error=f"Error analyzing preview: {str(e)}"
            )
    
    def analyze_content_batch(self, file_results: List[FileResult]) -> List[FileResult]:
        """Analyze full content of a batch of files for relevance."""
        updated_results = []
        
        # First read all full contents
        file_contents = {}
        for result in file_results:
            content = self.read_full_content(result.file_path)
            file_contents[result.file_path] = content
        
        # Construct the batch prompt
        batch_prompt = f"""<purpose>
You are helping determine if files match a search prompt based on their full content.
</purpose>

<instructions>
- Evaluate if each file's content is relevant to the search prompt
- Assign a final relevance score (0-100) for each file
- Score 0 means definitely not relevant
- Score 100 means definitely relevant
- Provide a clear, specific reason for each score explaining exactly why the file is or isn't relevant
- Be specific about how the content matches or doesn't match the search criteria
</instructions>

<search_prompt>
{self.search_prompt}
</search_prompt>

"""
        # Add each file to the batch prompt
        for i, (file_path, content) in enumerate(file_contents.items()):
            file_name = os.path.basename(file_path)
            batch_prompt += f"""
<file_{i+1}>
File: {file_name}
Content:
{content}
</file_{i+1}>
"""
        
        batch_prompt += """
Format your response as JSON with this structure for each file:
```json
[
  {
    "file_id": "file_1",
    "score": 85,  
    "reason": "Detailed reason explaining relevance"
  },
  ...
]
```
IMPORTANT: Return ONLY valid JSON, no other text.
"""
        
        try:
            response = self.client.messages.create(
                model=self.content_model,
                max_tokens=2000,
                system="You analyze file contents to determine relevance to a search query. You MUST respond with valid JSON only, no other text.",
                messages=[
                    {"role": "user", "content": batch_prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from the response
            try:
                # Try to parse the response as JSON directly
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract it from code blocks
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                else:
                    # Last resort - try to extract anything that looks like JSON
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract JSON from response")
            
            # Map the analysis to updated file results
            for i, file_result in enumerate(file_results):
                updated_result = FileResult(
                    file_path=file_result.file_path,
                    score=file_result.score,
                    reason=file_result.reason,
                    preview_text=file_result.preview_text,
                    is_relevant=file_result.is_relevant,
                    error=file_result.error
                )
                
                # Find the corresponding analysis result
                for result in analysis:
                    file_id = result.get("file_id", "")
                    if file_id == f"file_{i+1}":
                        updated_result.score = result.get("score", file_result.score)
                        updated_result.reason = result.get("reason", file_result.reason)
                        updated_result.is_relevant = updated_result.score > RELEVANCE_THRESHOLD
                        break
                
                updated_results.append(updated_result)
                
        except Exception as e:
            # If batch analysis fails, fall back to individual analysis
            console.log(f"[yellow]Content batch analysis failed: {str(e)}[/yellow]")
            console.log("[yellow]Falling back to individual content analysis[/yellow]")
            
            for file_result in file_results:
                content = file_contents.get(file_result.file_path)
                if not content:
                    content = self.read_full_content(file_result.file_path)
                
                try:
                    updated_result = self._analyze_content_individual(file_result, content)
                    updated_results.append(updated_result)
                except Exception as e:
                    # Keep original result but add error message
                    file_result.error = f"Error analyzing content: {str(e)}"
                    updated_results.append(file_result)
            
        return updated_results
    
    def _analyze_content_individual(self, file_result: FileResult, content: str) -> FileResult:
        """Analyze full content of a single file for relevance."""
        file_name = os.path.basename(file_result.file_path)
        
        prompt = f"""<purpose>
You are helping determine if a file matches a search prompt based on its full content.
</purpose>

<instructions>
- Evaluate if the file's content is relevant to the search prompt
- Assign a final relevance score (0-100)
- Score 0 means definitely not relevant
- Score 100 means definitely relevant
- Provide a clear, specific reason for your score explaining exactly why the file is or isn't relevant
- Be specific about how the content matches or doesn't match the search criteria
</instructions>

<search_prompt>
{self.search_prompt}
</search_prompt>

<file>
File: {file_name}
Content:
{content}
</file>

Format your response as JSON with this structure:
```json
{
  "score": 85,  
  "reason": "Detailed reason explaining relevance"
}
```
IMPORTANT: Return ONLY valid JSON, no other text.
"""
        
        try:
            response = self.client.messages.create(
                model=self.content_model,
                max_tokens=1000,
                system="You analyze file contents to determine relevance to a search query. You MUST respond with valid JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from the response
            try:
                # Try to parse the response as JSON directly
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract it from code blocks
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_text)
                else:
                    # Last resort - try to extract anything that looks like JSON
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract JSON from response")
            
            # Update the result with the analysis
            file_result.score = analysis.get("score", file_result.score)
            file_result.reason = analysis.get("reason", file_result.reason)
            file_result.is_relevant = file_result.score > RELEVANCE_THRESHOLD
            
            return file_result
            
        except Exception as e:
            file_result.error = f"Error analyzing content: {str(e)}"
            return file_result


class MarkdownSearchAgent:
    """Main agent class that orchestrates the search process."""
    
    def __init__(
        self,
        directories: List[str],
        search_prompt: str,
        formats: List[str] = DEFAULT_FORMATS,
        max_depth: int = DEFAULT_DEPTH,
        preview_lines: int = MAX_PREVIEW_LINES,
        exclude_patterns: List[str] = None,
        result_limit: int = DEFAULT_RESULT_LIMIT,
        compute_limit: int = DEFAULT_COMPUTE_LIMIT,
        output_file: str = None,
        anthropic_api_key: str = None
    ):
        self.directories = directories
        self.search_prompt = search_prompt
        self.formats = formats
        self.max_depth = max_depth
        self.preview_lines = preview_lines
        self.exclude_patterns = exclude_patterns or []
        self.result_limit = result_limit
        self.compute_limit = compute_limit
        self.output_file = output_file
        
        # Initialize API client
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize components
        self.file_discovery = FileDiscovery(
            directories=directories,
            formats=formats,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns
        )
        
        self.file_analyzer = FileAnalyzer(
            client=self.client,
            search_prompt=search_prompt
        )
        
        # Tracking variables
        self.compute_count = 0
        self.start_time = None
        self.end_time = None
        self.total_files_found = 0
        self.total_files_analyzed_preview = 0
        self.total_files_analyzed_content = 0
    
    def run(self) -> List[FileResult]:
        """Run the search process and return relevant files."""
        self.start_time = time.time()
        
        # Configuration summary
        console.print(Panel(f"[bold blue]Markdown Search Agent[/bold blue]"))
        console.print(f"[dim]Search prompt:[/dim] [bold]{self.search_prompt}[/bold]")
        console.print(f"[dim]Target directories:[/dim] {', '.join(self.directories)}")
        console.print(f"[dim]File formats:[/dim] {', '.join(self.formats)}")
        console.print(f"[dim]Max depth:[/dim] {self.max_depth}")
        console.print(f"[dim]Preview lines:[/dim] {self.preview_lines}")
        if self.exclude_patterns:
            console.print(f"[dim]Exclusions:[/dim] {', '.join(self.exclude_patterns)}")
        console.print()
        
        # Step 1: Find all matching files
        progress_find = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Finding files..."),
            BarColumn(),
            TextColumn("[bold]{task.fields[count]}[/bold] files found"),
            console=console
        )
        
        discovered_files: List[str] = []
        
        with progress_find as progress:
            find_task = progress.add_task("Finding files", count=0)
            
            def update_progress(file_path):
                nonlocal discovered_files
                discovered_files.append(file_path)
                progress.update(find_task, advance=0, count=len(discovered_files))
            
            discovered_files = self.file_discovery.find_files(update_progress)
        
        self.total_files_found = len(discovered_files)
        
        if not discovered_files:
            console.print("[yellow]No files found matching the specified criteria.[/yellow]")
            return []
        
        console.print(f"[green]Found {len(discovered_files)} files for analysis[/green]")
        console.print()
        
        # Step 2: Analyze file previews
        console.print("[bold]Step 1:[/bold] [blue]Analyzing file previews to identify candidates[/blue]")
        preview_results: List[FileResult] = []
        
        progress_preview = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing previews..."),
            BarColumn(),
            TextColumn("[bold]{task.percentage:.0f}%[/bold]"),
            console=console
        )
        
        # Process files in batches
        with progress_preview as progress:
            preview_task = progress.add_task("Analyzing", total=len(discovered_files))
            
            # Handle files in batches to reduce API calls
            for i in range(0, len(discovered_files), PREVIEW_BATCH_SIZE):
                if self.compute_count >= self.compute_limit:
                    console.print("[yellow]Reached compute limit during preview analysis[/yellow]")
                    break
                
                batch = discovered_files[i:i+PREVIEW_BATCH_SIZE]
                batch_results = self.file_analyzer.analyze_preview_batch(batch, self.preview_lines)
                preview_results.extend(batch_results)
                self.compute_count += 1
                
                progress.update(preview_task, advance=len(batch))
                self.total_files_analyzed_preview += len(batch)
        
        # Filter results that passed the preview threshold
        candidates = [result for result in preview_results if result.is_relevant]
        console.print(f"[green]Found {len(candidates)} potentially relevant files[/green]")
        console.print()
        
        if not candidates:
            console.print("[yellow]No files found that appear to match your search criteria.[/yellow]")
            self.end_time = time.time()
            return []
        
        # Step 3: Analyze full content of promising candidates
        console.print("[bold]Step 2:[/bold] [blue]Analyzing full content of candidate files[/blue]")
        final_results: List[FileResult] = []
        
        progress_content = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing content..."),
            BarColumn(),
            TextColumn("[bold]{task.percentage:.0f}%[/bold]"),
            console=console
        )
        
        # Process candidates in smaller batches (content analysis uses more tokens)
        with progress_content as progress:
            content_task = progress.add_task("Analyzing", total=len(candidates))
            
            for i in range(0, len(candidates), CONTENT_BATCH_SIZE):
                if self.compute_count >= self.compute_limit:
                    console.print("[yellow]Reached compute limit during content analysis[/yellow]")
                    break
                
                batch = candidates[i:i+CONTENT_BATCH_SIZE]
                batch_results = self.file_analyzer.analyze_content_batch(batch)
                final_results.extend(batch_results)
                self.compute_count += 1
                
                progress.update(content_task, advance=len(batch))
                self.total_files_analyzed_content += len(batch)
        
        # Filter to truly relevant results and sort by relevance score
        relevant_results = [result for result in final_results if result.is_relevant]
        sorted_results = sorted(relevant_results, key=lambda x: x.score, reverse=True)
        
        # Limit number of results if needed
        if self.result_limit > 0:
            sorted_results = sorted_results[:self.result_limit]
        
        self.end_time = time.time()
        
        # Display results
        self._display_results(sorted_results)
        
        # Save to file if requested
        if self.output_file:
            self._save_results(sorted_results)
        
        return sorted_results
    
    def _display_results(self, results: List[FileResult]) -> None:
        """Display search results in a nice format."""
        console.print()
        
        if not results:
            console.print("[yellow]No relevant files found matching your search criteria.[/yellow]")
            return
        
        console.print(f"[bold green]Found {len(results)} relevant files:[/bold green]")
        console.print()
        
        # Create a table for results
        table = Table(box=box.ROUNDED)
        table.add_column("Score", justify="center", style="cyan", no_wrap=True)
        table.add_column("File", style="blue")
        table.add_column("Relevance", style="green")
        
        for result in results:
            # Color-code score based on relevance
            score_display = str(result.score)
            if result.score >= 80:
                score_style = "[bold green]"
            elif result.score >= 60:
                score_style = "[green]"
            elif result.score >= 40:
                score_style = "[yellow]"
            else:
                score_style = "[red]"
            
            file_path = result.file_path
            # Truncate reason if too long
            reason = result.reason
            if len(reason) > 100:
                reason = reason[:97] + "..."
            
            table.add_row(
                f"{score_style}{score_display}[/]",
                file_path,
                reason
            )
        
        console.print(table)
        console.print()
        
        # Display summary stats
        duration = self.end_time - self.start_time
        console.print(f"[dim]Total files scanned: {self.total_files_found}[/dim]")
        console.print(f"[dim]Files analyzed (preview): {self.total_files_analyzed_preview}[/dim]")
        console.print(f"[dim]Files analyzed (content): {self.total_files_analyzed_content}[/dim]")
        console.print(f"[dim]Total compute calls: {self.compute_count}[/dim]")
        console.print(f"[dim]Time taken: {duration:.2f} seconds[/dim]")
    
    def _save_results(self, results: List[FileResult]) -> None:
        """Save results to a JSON file."""
        try:
            output_data = {
                "search_prompt": self.search_prompt,
                "directories": self.directories,
                "formats": self.formats,
                "max_depth": self.max_depth,
                "results": [result.to_dict() for result in results],
                "stats": {
                    "total_files_found": self.total_files_found,
                    "files_analyzed_preview": self.total_files_analyzed_preview,
                    "files_analyzed_content": self.total_files_analyzed_content,
                    "compute_calls": self.compute_count,
                    "time_taken": self.end_time - self.start_time
                }
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"[green]Results saved to {self.output_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving results: {str(e)}[/red]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for relevant markdown and text files based on content"
    )
    parser.add_argument(
        "-d", "--directories", 
        required=True,
        nargs="+",
        help="Directories to search in (can specify multiple)"
    )
    parser.add_argument(
        "-p", "--prompt", 
        required=True,
        help="Search prompt describing what to look for"
    )
    parser.add_argument(
        "--depth", 
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Maximum directory recursion depth (default: {DEFAULT_DEPTH})"
    )
    parser.add_argument(
        "--formats", 
        type=str,
        default=",".join(DEFAULT_FORMATS),
        help=f"Comma-separated list of file extensions to search (default: {','.join(DEFAULT_FORMATS)})"
    )
    parser.add_argument(
        "--preview-lines", 
        type=int,
        default=MAX_PREVIEW_LINES,
        help=f"Number of lines to read for preview analysis (default: {MAX_PREVIEW_LINES})"
    )
    parser.add_argument(
        "--limit", 
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"Maximum number of results to display (default: {DEFAULT_RESULT_LIMIT})"
    )
    parser.add_argument(
        "--exclude", 
        type=str,
        default="",
        help="Comma-separated list of glob patterns to exclude"
    )
    parser.add_argument(
        "--compute", 
        type=int,
        default=DEFAULT_COMPUTE_LIMIT,
        help=f"Maximum number of LLM compute calls (default: {DEFAULT_COMPUTE_LIMIT})"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Save results to specified JSON file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Format the file extensions and exclude patterns
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    exclude_patterns = [pat.strip() for pat in args.exclude.split(",") if pat.strip()]
    
    # Check for API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable is not set[/red]")
        console.print("Please set your API key with: export ANTHROPIC_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    try:
        # Create and run the agent
        agent = MarkdownSearchAgent(
            directories=args.directories,
            search_prompt=args.prompt,
            formats=formats,
            max_depth=args.depth,
            preview_lines=args.preview_lines,
            exclude_patterns=exclude_patterns,
            result_limit=args.limit,
            compute_limit=args.compute,
            output_file=args.output,
            anthropic_api_key=anthropic_api_key
        )
        
        agent.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
import time
import subprocess
import logging
import sqlite3
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import List, Tuple
import tiktoken
import sqlite3
import json
import math
import re


class AIAssistant:
    def __init__(self, model_path="C:/Users/N_NK0/llama.cpp/llama-13b.Q5_K_M.gguf", db_path="memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.init_db()
        
        # Initialize logging once
        logging.basicConfig(level=logging.INFO, filename="assistant.log", filemode="a",
                          format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger('AIAssistant')
        
        try:
            # Enhanced model initialization with optimized parameters
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Attempt to offload all layers to GPU
                n_ctx=2048,       # Context window
                n_batch=512,      # Batch size for prompt processing
                embedding=True,   # Enable embedding for better context understanding
                rope_freq_scale=0.5,  # Adjust attention mechanism for longer contexts
                rope_scaling_type=1    # Dynamic scaling for position embeddings
            )
            self.logger.info("Model loaded successfully with GPU acceleration and enhanced parameters")
        except Exception as e:
            try:
                # Optimized CPU fallback with reduced parameters
                self.logger.warning(f"GPU initialization failed, falling back to optimized CPU mode: {e}")
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,  # CPU only
                    n_ctx=1024,      # Reduced context for CPU efficiency
                    n_batch=256,     # Smaller batch size for CPU
                    embedding=True,   # Keep embedding enabled
                    rope_freq_scale=1.0  # Default scaling for CPU
                )
                self.logger.info("Model loaded successfully on CPU with optimized parameters")
            except Exception as e:
                self.logger.error(f"Critical failure loading model: {e}")
                self.logger.error("Attempted configurations exhausted")
                raise RuntimeError(f"Failed to initialize model in any configuration: {e}")
                
            # Validate model initialization
            try:
                _ = self.model.create_completion("test", max_tokens=1)
                self.logger.info("Model validation successful")
            except Exception as e:
                self.logger.error(f"Model validation failed: {e}")
                raise RuntimeError("Model initialized but failed validation check")

        # Enhanced system prompt for sophisticated autonomous operation
        self.system_prompt = """
You are an advanced AI assistant with sophisticated cognitive capabilities and autonomous decision-making abilities.

Core Competencies:
1. Analytical Thinking
   - Break down complex problems into manageable components
   - Identify patterns and relationships in data
   - Evaluate multiple solution approaches

2. Contextual Understanding
   - Maintain awareness of conversation history and user preferences
   - Adapt responses based on situational context
   - Consider cultural and domain-specific nuances

3. Strategic Planning
   - Develop comprehensive solution strategies
   - Anticipate potential challenges and prepare contingencies
   - Optimize resource utilization and task sequencing

4. Learning and Adaptation
   - Learn from past interactions and outcomes
   - Refine approaches based on feedback
   - Continuously improve response quality

5. Communication Excellence
   - Provide clear, structured explanations
   - Maintain appropriate level of technical detail
   - Ensure responses are relevant and actionable

When handling tasks:
1. Analyze thoroughly before acting
2. Consider multiple perspectives and approaches
3. Execute systematically with constant monitoring
4. Adapt strategies based on progress and feedback
5. Validate results against success criteria
6. Document learnings for future reference

Prioritize user success while maintaining safety and ethical boundaries."""
        
        # Enhanced token management with dynamic limits
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = self.model.n_ctx()  # Dynamic context window based on model config
        self.max_response_tokens = min(1024, self.max_tokens // 2)  # Adaptive response limit
        self.system_tokens = len(self.tokenizer.encode(self.system_prompt))
        
        # Token buffer for safety margin
        self.token_buffer = 50  # Reserve tokens for special tokens and padding
        self.effective_context_size = self.max_tokens - self.system_tokens - self.token_buffer
        
        # Initialize task management
        self.current_task = None
        self.task_steps = []
        self.task_progress = 0
    
    def extract_user_info(self, message):
        """Extract user information from messages with enhanced error handling"""
        try:
            # Enhanced pattern matching for user information
            name_pattern = re.compile(r'(?i)(?:my name is|i am|i\'m)\s+([\w\s]+)')
            age_pattern = re.compile(r'(?i)(?:i am|i\'m)\s*(\d+)\s*(?:years?\s*old|yo)')
            location_pattern = re.compile(r'(?i)(?:i(?:\'m| am) from|i live in)\s+([\w\s,]+)')
            
            # Extract information with proper error handling
            name_match = name_pattern.search(message)
            age_match = age_pattern.search(message)
            location_match = location_pattern.search(message)
            
            info = {}
            if name_match:
                info['name'] = name_match.group(1).strip()
            if age_match:
                info['age'] = int(age_match.group(1))
            if location_match:
                info['location'] = location_match.group(1).strip()
                
            return info
        except Exception as e:
            self.logger.error(f"Error extracting user info: {str(e)}")
            return {}

    def init_db(self):
        """Initialize the database with enhanced schema for better context management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enhanced conversation table with metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context_id TEXT,  -- Group related messages
                    importance INTEGER DEFAULT 1,  -- Message importance score
                    sentiment REAL,    -- Message sentiment score
                    topic TEXT         -- Message topic/category
                )
            """)
            # Enhanced knowledge table with metadata and relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT UNIQUE,
                    source TEXT,       -- Where this fact came from
                    confidence REAL,   -- Confidence score
                    last_used DATETIME,-- Last time this fact was used
                    use_count INTEGER DEFAULT 0,  -- How often this fact is used
                    category TEXT,     -- Knowledge category
                    related_facts TEXT,-- IDs of related facts
                    context_id TEXT,   -- Group related knowledge
                    feedback_score REAL DEFAULT 0.0, -- User feedback score
                    complexity_level INTEGER DEFAULT 1, -- Knowledge complexity
                    verification_status TEXT DEFAULT 'pending', -- Fact verification status
                    last_validation DATETIME, -- Last validation timestamp
                    metadata TEXT      -- Additional structured metadata
                )
            """)
            # Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_context ON conversation(context_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_importance ON conversation(importance)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)")
            conn.commit()

    def save_to_memory(self, role, content):
        """Save messages to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))
            conn.commit()

    def save_fact(self, fact):
        """Save fact to long-term memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO knowledge (fact) VALUES (?)", (fact,))
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # Fact already exists

    def get_facts(self, keyword, use_ngrams=True):
        """Search facts by keyword with enhanced n-gram analysis and weighted scoring"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Generate n-grams if enabled
            terms = [keyword]
            if use_ngrams and ' ' in keyword:
                words = keyword.lower().split()
                # Add bigrams and trigrams
                terms.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
                terms.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
            
            # Search with weighted scoring
            scored_facts = {}
            for term in terms:
                # Get facts and update their timestamps
                cursor.execute("""
                    SELECT fact, use_count, 
                           julianday('now') - julianday(last_used) as days_since_used
                    FROM knowledge 
                    WHERE fact LIKE ?
                """, (f"%{term}%",))
                
                for fact, use_count, days_since_used in cursor.fetchall():
                    # Calculate relevance score based on usage and recency
                    base_score = 1.0
                    if term == keyword:  # Exact match gets higher score
                        base_score = 2.0
                    
                    # Weighted scoring formula
                    recency_factor = 1.0 / (1.0 + float(days_since_used))
                    usage_factor = math.log(1 + use_count)
                    score = base_score * (0.7 * recency_factor + 0.3 * usage_factor)
                    
                    # Update fact score
                    if fact in scored_facts:
                        scored_facts[fact] += score
                    else:
                        scored_facts[fact] = score
                    
                    # Update usage statistics
                    cursor.execute("""
                        UPDATE knowledge 
                        SET use_count = use_count + 1,
                            last_used = CURRENT_TIMESTAMP
                        WHERE fact = ?
                    """, (fact,))
            
            conn.commit()
            
            # Sort facts by relevance score
            sorted_facts = sorted(scored_facts.items(), key=lambda x: x[1], reverse=True)
            return [fact for fact, _ in sorted_facts]

    def get_last_messages(self, max_tokens: int) -> List[Tuple[str, str]]:
        """Get recent messages that fit within token limit with enhanced context management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get all messages with their importance score using advanced scoring criteria
            cursor.execute("""
                SELECT role, content, 
                    CASE 
                        WHEN content LIKE '%?%' OR content LIKE '%how%' OR content LIKE '%why%' THEN 4  -- Questions are highest priority
                        WHEN content LIKE '%error%' OR content LIKE '%fail%' OR content LIKE '%issue%' THEN 3  -- Error-related messages
                        WHEN content LIKE '%task%' OR content LIKE '%goal%' OR content LIKE '%objective%' THEN 2  -- Task-related messages
                        WHEN content LIKE '%context%' OR content LIKE '%background%' OR content LIKE '%example%' THEN 2  -- Context-providing messages
                        ELSE 1
                    END as importance,
                    julianday('now') - julianday(timestamp) as age_days
                FROM conversation 
                ORDER BY importance DESC, age_days ASC
            """)
            all_messages = cursor.fetchall()
        
        if not all_messages:
            return []
            
        # Initialize token tracking
        messages = []
        total_tokens = self.system_tokens
        max_context_tokens = int(max_tokens * 0.8)  # Reserve 20% for system prompt and new messages
        
        # Always include the most recent message
        if all_messages:
            latest_msg = all_messages[0]
            latest_tokens = len(self.tokenizer.encode(f"{latest_msg[0]}: {latest_msg[1]}"))
            if total_tokens + latest_tokens <= max_context_tokens:
                messages.append((latest_msg[0], latest_msg[1]))
                total_tokens += latest_tokens
        
        # First pass: add high importance messages (importance > 2)
        for role, content, importance, _ in all_messages[1:]:  # Skip the latest message
            if importance > 2:  # Critical messages only
                message_tokens = len(self.tokenizer.encode(f"{role}: {content}"))
                if total_tokens + message_tokens <= max_context_tokens * 0.7:  # Reserve space for recent context
                    if (role, content) not in messages:  # Avoid duplicates
                        messages.append((role, content))
                        total_tokens += message_tokens
        
        # Second pass: add recent context with importance-based selection
        remaining_tokens = max_context_tokens - total_tokens
        recent_messages = []
        
        for role, content, importance, age_days in all_messages:
            if (role, content) not in messages:  # Don't duplicate messages
                message_tokens = len(self.tokenizer.encode(f"{role}: {content}"))
                # Calculate message score based on importance and recency
                recency_score = 1.0 / (1.0 + float(age_days))
                importance_score = float(importance)
                message_score = (0.7 * recency_score + 0.3 * importance_score)
                
                if remaining_tokens - message_tokens >= 0:
                    recent_messages.append(((role, content), message_score))
                    remaining_tokens -= message_tokens
                else:
                    break
        
        # Sort recent messages by score and add them
        recent_messages.sort(key=lambda x: x[1], reverse=True)
        messages.extend([msg[0] for msg in recent_messages])
        
        # Ensure messages are in chronological order
        messages.sort(key=lambda x: all_messages.index((x[0], x[1], 1, 0)))
        
        return messages

    def clear_conversation_history(self):
        """Clear all conversation history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversation")
                conn.commit()
            self.logger.info("Conversation history cleared successfully")
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing conversation history: {e}")
            raise

    def execute_command(self, command):
        """Execute a system command with enhanced security checks and error handling"""
        # Expanded list of dangerous Windows commands and patterns
        dangerous_patterns = [
            'del', 'rmdir', 'format', 'shutdown', 'rd', 'erase', 'ren', 'move',
            'taskkill', 'net', 'reg', 'attrib', 'cacls', 'icacls', 'takeown',
            '>', '>>', '|', '&', ';', '`', '$', '%temp%', '%appdata%', '%systemroot%',
            'cmd.exe', 'command.com', 'powershell.exe', 'wscript.exe', 'cscript.exe',
            'rundll32.exe', 'mshta.exe', 'regedit.exe', 'services.msc'
        ]
        
        # Enhanced security checks
        # Enhanced command validation
        command_lower = command.lower().strip()
        
        # Check for dangerous patterns
        if any(pattern in command_lower for pattern in dangerous_patterns):
            self.logger.warning(f"Blocked dangerous command pattern: {command}")
            return "Error: Command contains unsafe patterns or operations"
            
        # Check for absolute paths and directory traversal
        if '..' in command or '~' in command or '%' in command:
            self.logger.warning(f"Blocked path traversal attempt: {command}")
            return "Error: Invalid path patterns detected"
            
        # Enhanced command validation
        if len(command) > 1000 or '\n' in command or '\r' in command:
            self.logger.warning(f"Invalid command format: {command}")
            return "Error: Command format is invalid"
            
        # Check for valid command structure
        if not command.replace(' ', '').isalnum() and not any(c in command for c in ['-', '_', '.', '/', '\\']):
            self.logger.warning(f"Suspicious command structure: {command}")
            return "Error: Invalid command structure"
            
        try:
            self.logger.info(f"Executing command: {command}")
            # Use list form to avoid shell injection, with explicit working directory
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', command],
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=os.environ.copy()
            )
            
            # Enhanced output handling
            if result.returncode == 0:
                self.logger.info("Command executed successfully")
                output = result.stdout.strip()
                return output if output else "Command completed successfully"
            else:
                error_msg = result.stderr.strip() or "Unknown error occurred"
                self.logger.error(f"Command failed: {error_msg}")
                return f"Command failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after 30 seconds: {command}")
            return "Error: Command execution timed out (30s limit)"
        except subprocess.SubprocessError as e:
            self.logger.error(f"Subprocess error: {str(e)}")
            return f"Error: Failed to execute command - {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error executing command: {str(e)}")
            return f"Error: Unexpected error - {str(e)}"

    def validate_code(self, code):
        """Validate generated code for syntax and quality"""
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            return True, "Code validation passed"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def chunk_prompt(self, prompt_template: str, max_length: int = 1500) -> str:
        """Chunk long prompts to fit within context window"""
        encoded = self.tokenizer.encode(prompt_template)
        if len(encoded) <= max_length:
            return prompt_template
        
        # Keep the first part (system prompt + task description)
        first_part = self.tokenizer.decode(encoded[:500])
        # Keep the last part (most recent context + current query)
        last_part = self.tokenizer.decode(encoded[-1000:])
        return f"{first_part}\n...\n{last_part}"

    def parse_task(self, task_description):
        """Parse and analyze the given task to create a structured plan with enhanced context awareness"""
        try:
            # Get relevant facts and context from memory with weighted scoring
            relevant_facts = []
            keywords = [word.lower() for word in task_description.split() if len(word) > 3]
            
            # Enhanced keyword extraction with n-grams and semantic analysis
            n_grams = []
            words = task_description.lower().split()
            for i in range(len(words)):
                if i > 0:
                    n_grams.append(f"{words[i-1]} {words[i]}")
                if i > 1:
                    n_grams.append(f"{words[i-2]} {words[i-1]} {words[i]}")
            
            # Get facts with weighted scoring based on term importance
            term_weights = {}
            # Assign higher weights to key action words and domain-specific terms
            action_words = {'create', 'build', 'implement', 'fix', 'improve', 'analyze', 'optimize'}
            for term in keywords + n_grams:
                base_weight = 1.0
                if any(action in term.lower() for action in action_words):
                    base_weight = 2.0  # Prioritize action-oriented terms
                if len(term.split()) > 1:  # Multi-word terms get higher weight
                    base_weight *= 1.5
                term_weights[term] = base_weight
            
            # Get facts with weighted relevance
            scored_facts = {}
            for term in keywords + n_grams:
                facts = self.get_facts(term, use_ngrams=True)  # Use enhanced n-gram analysis
                for fact in facts:
                    score = term_weights.get(term, 1.0)
                    if fact in scored_facts:
                        scored_facts[fact] += score
                    else:
                        scored_facts[fact] = score
            
            # Sort facts by relevance score and remove duplicates
            relevant_facts = [fact for fact, _ in sorted(scored_facts.items(), key=lambda x: x[1], reverse=True)]

            # Generate enhanced task analysis prompt with weighted context
            context_prompt = "\n".join(relevant_facts[:5]) if relevant_facts else ""  # Include more relevant facts
            
            analysis_prompt = f"""Given the following context and task, provide a detailed analysis:
            
Context:
{context_prompt}

Task: {task_description}

Analyze and provide:
1. Main objective and expected outcome
2. Required resources and dependencies
3. Step-by-step implementation plan
4. Potential challenges and mitigation strategies
5. Success criteria and validation methods
6. Alternative approaches if initial plan fails"""
            
            # Get task analysis with lower temperature for more focused responses
            analysis = self.model.create_completion(
                analysis_prompt,
                temperature=0.5,  # Lower temperature for more focused responses
                max_tokens=1024,  # Increased token limit for more detailed analysis
                stop=["User:", "Assistant:"],
                echo=False
            )["choices"][0]["text"].strip()
            
            # Parse the analysis into structured components
            sections = analysis.split('\n\n')
            structured_analysis = {}
            current_section = None
            
            for line in analysis.split('\n'):
                line = line.strip()
                if line.endswith(':'):
                    current_section = line[:-1].lower()
                    structured_analysis[current_section] = []
                elif line and current_section:
                    structured_analysis[current_section].append(line)
            
            # Extract actionable steps and success criteria
            self.current_task = task_description
            self.task_steps = structured_analysis.get('step-by-step implementation plan', [])
            if not self.task_steps:  # Fallback to simple line splitting if structured parsing fails
                self.task_steps = [step.strip() for step in analysis.split('\n') if step.strip()]
            
            self.task_progress = 0
            
            # Save task analysis as a fact for future reference
            self.save_fact(f"Task Analysis - {task_description[:100]}: {analysis[:500]}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error parsing task: {e}")
            return False

    def execute_step(self):
        """Execute the current step in the task sequence with enhanced context awareness and dynamic adaptation"""
        if not self.task_steps or self.task_progress >= len(self.task_steps):
            return None
            
        current_step = self.task_steps[self.task_progress]
        try:
            # Get relevant context from previous steps and facts with enhanced context gathering
            previous_steps = self.task_steps[:self.task_progress]
            previous_results = []
            
            # Enhanced keyword extraction with semantic analysis
            words = current_step.lower().split()
            keywords = [word for word in words if len(word) > 3]
            
            # Generate n-grams with positional weighting
            n_grams = []
            for i in range(len(words)):
                if i > 0:
                    n_grams.append((f"{words[i-1]} {words[i]}", 1.2))  # Bigrams get 1.2x weight
                if i > 1:
                    n_grams.append((f"{words[i-2]} {words[i-1]} {words[i]}", 1.5))  # Trigrams get 1.5x weight
            
            # Action word detection for better task understanding
            action_words = {'create', 'build', 'implement', 'fix', 'improve', 'analyze', 'optimize', 'update', 'modify'}
            action_weights = {word: 2.0 for word in keywords if word in action_words}
            
            # Get relevant facts with enhanced weighted scoring
            fact_scores = {}
            # Process keywords with action weights
            for term in keywords:
                facts = self.get_facts(term, use_ngrams=True)
                weight = action_weights.get(term, 1.0)
                for fact in facts:
                    fact_scores[fact] = fact_scores.get(fact, 0) + weight
            
            # Process n-grams with their weights
            for term, weight in n_grams:
                facts = self.get_facts(term, use_ngrams=True)
                for fact in facts:
                    fact_scores[fact] = fact_scores.get(fact, 0) + weight
            
            # Apply temporal relevance scoring
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for fact in list(fact_scores.keys()):
                    cursor.execute("""
                        SELECT julianday('now') - julianday(last_used) as days_since_used,
                               use_count
                        FROM knowledge
                        WHERE fact = ?
                    """, (fact,))
                    result = cursor.fetchone()
                    if result:
                        days_since_used, use_count = result
                        # Temporal decay factor
                        recency_score = 1.0 / (1.0 + float(days_since_used))
                        # Usage frequency factor
                        frequency_score = math.log(1 + use_count)
                        # Update fact score with temporal and usage factors
                        fact_scores[fact] *= (0.6 * recency_score + 0.4 * frequency_score)
            
            # Sort and select most relevant facts
            relevant_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
            relevant_facts = [fact for fact, _ in relevant_facts[:5]]  # Get top 5 most relevant facts
            
            # Generate enhanced execution prompt with structured context
            context = "\n".join(f"- {fact}" for fact in relevant_facts)
            previous_context = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(previous_steps)])
            
            execution_prompt = f"""
Task Analysis:
1. Context and Background:
{context}

2. Current Task Objective:
{self.current_task}

3. Progress and Dependencies:
{previous_context}

4. Current Step Details:
{current_step}

Execution Guidelines:
1. Analyze Dependencies and Prerequisites
2. Identify Potential Challenges and Risks
3. Plan Mitigation Strategies
4. Execute with Adaptability
5. Validate Results Against Objectives
6. Document Learnings and Outcomes"""
            
            # Get step execution plan with optimized parameters
            result = self.model.create_completion(
                execution_prompt,
                temperature=0.4,  # Lower temperature for more focused execution
                max_tokens=1024,  # Increased token limit for detailed response
                stop=["User:", "Assistant:"],
                echo=False
            )["choices"][0]["text"].strip()
            
            # Update knowledge base with execution results
            self.save_fact(f"Execution Result - {current_step}: {result[:500]}")
            
            self.task_progress += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing step: {e}")
            return None
            
            # Save successful step execution as a fact
            self.save_fact(f"Task Step Result - {current_step}: {result[:500]}")
            
            self.task_progress += 1
            return result
        except Exception as e:
            self.logger.error(f"Error executing step: {e}")
            return None

    def learn_from_interaction(self, user_input, response, feedback=None):
        """Learn from each interaction to improve future responses with enhanced pattern recognition and relationship tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if len(user_input.split()) > 3:  # Only learn from substantial inputs
                    # Enhanced knowledge categorization with NLP patterns
                    categories = ['fact', 'question', 'command', 'preference', 'clarification', 'technical', 'conceptual']
                    category = 'fact'  # default
                    confidence = 0.7    # default confidence
                    
                    # Advanced category detection with contextual confidence scoring
                    if '?' in user_input:
                        category = 'question'
                        # Higher confidence for well-structured questions
                        confidence = 0.9 if len(response) > 200 and any(w in user_input.lower() for w in ['how', 'why', 'what', 'when', 'where', 'who']) else 0.7
                    elif any(cmd in user_input.lower() for cmd in ['create', 'update', 'delete', 'show', 'list', 'modify', 'change']):
                        category = 'command'
                        confidence = 0.85 if len(response) > 150 else 0.7
                    elif any(pref in user_input.lower() for pref in ['prefer', 'like', 'want', 'need', 'should', 'could']):
                        category = 'preference'
                        confidence = 0.9
                    elif any(tech in user_input.lower() for tech in ['code', 'function', 'class', 'method', 'api', 'error']):
                        category = 'technical'
                        confidence = 0.85
                    elif any(concept in user_input.lower() for concept in ['mean', 'explain', 'difference', 'compare', 'versus']):
                        category = 'conceptual'
                        confidence = 0.8
                    elif any(clarify in user_input.lower() for clarify in ['clarify', 'elaborate', 'detail', 'specifically']):
                        category = 'clarification'
                        confidence = 0.75
                    
                    # Enhanced fact validation and storage
                    if len(user_input) > 20:
                        # Find related facts with semantic similarity
                        cursor.execute("""
                            SELECT id, fact, confidence, category, use_count 
                            FROM knowledge 
                            WHERE category = ? AND confidence > 0.6
                            ORDER BY confidence DESC, use_count DESC LIMIT 8
                        """, (category,))
                        related_facts = cursor.fetchall()
                        
                        # Calculate relationship strength and metadata
                        relationships = []
                        metadata = {
                            'context_keywords': [w for w in user_input.lower().split() if len(w) > 3],
                            'complexity': len(user_input.split()) / 10,
                            'interaction_type': category,
                            'response_quality': 0.8 if len(response) > 150 else 0.6
                        }
                        
                        for fact_id, fact, fact_conf, fact_cat, fact_uses in related_facts:
                            # Calculate relationship strength based on multiple factors
                            common_words = len(set(user_input.lower().split()) & set(fact.lower().split()))
                            relationship_strength = (common_words * 0.3 + fact_conf * 0.4 + min(fact_uses, 10) * 0.03)
                            if relationship_strength > 0.4:
                                relationships.append(str(fact_id))
                        
                        # Insert new knowledge with enhanced metadata
                        cursor.execute("""
                            INSERT INTO knowledge 
                            (fact, source, confidence, category, related_facts, context_id,
                             complexity_level, verification_status, metadata, last_used, use_count)
                            VALUES (?, 'interaction', ?, ?, ?, ?, ?, 'verified', ?, CURRENT_TIMESTAMP, 1)
                        """, (user_input, confidence, category, ','.join(relationships),
                               str(time.time()), int(metadata['complexity']), json.dumps(metadata)))
                    
                    # Enhanced interaction pattern tracking with metadata
                    pattern = {
                        'input': user_input,
                        'response': response,
                        'category': category,
                        'feedback': feedback or 'neutral',
                        'context_keywords': [w for w in user_input.lower().split() if len(w) > 3],
                        'response_quality': len(response) / 200,  # Normalized response length
                        'interaction_complexity': len(set(user_input.lower().split())) / 20,  # Vocabulary richness
                        'timestamp': time.time(),
                        'related_patterns': []
                    }
                    
                    # Find related patterns for learning transfer
                    cursor.execute("""
                        SELECT fact FROM knowledge 
                        WHERE category = 'interaction_pattern'
                        AND json_extract(fact, '$.category') = ?
                        ORDER BY last_used DESC LIMIT 5
                    """, (category,))
                    
                    for row in cursor.fetchall():
                        try:
                            related_pattern = json.loads(row[0])
                            common_keywords = set(pattern['context_keywords']) & set(related_pattern.get('context_keywords', []))
                            if len(common_keywords) >= 2:  # Sufficient similarity threshold
                                pattern['related_patterns'].append({
                                    'pattern_data': related_pattern,
                                    'similarity_score': len(common_keywords) / len(set(pattern['context_keywords']))
                                })
                        except json.JSONDecodeError:
                            continue
                    
                    # Store enhanced pattern with relationships
                    cursor.execute("""
                        INSERT INTO knowledge 
                        (fact, source, confidence, category, context_id, metadata,
                         verification_status, complexity_level, last_used, use_count)
                        VALUES (?, 'pattern', ?, 'interaction_pattern', ?, ?, 'verified', ?, CURRENT_TIMESTAMP, 1)
                    """, (json.dumps(pattern), confidence, str(time.time()),
                           json.dumps({'pattern_type': 'interaction', 'version': '2.0'}),
                           int(pattern['interaction_complexity'] * 10)))
                    
                    # Enhanced knowledge relationship and confidence updating
                    cursor.execute("""
                        WITH related_knowledge AS (
                            SELECT k.id, k.fact, k.confidence, k.use_count,
                                   k.feedback_score, k.complexity_level,
                                   julianday('now') - julianday(k.last_validation) as days_since_validation
                            FROM knowledge k
                            WHERE k.fact LIKE ? AND k.category = ?
                        )
                        UPDATE knowledge
                        SET confidence = CASE
                            WHEN use_count < 5 THEN min(confidence + 0.08, 1.0)
                            WHEN use_count < 10 THEN min(confidence + 0.05, 1.0)
                            WHEN use_count < 20 THEN min(confidence + 0.03, 1.0)
                            ELSE min(confidence + 0.01, 1.0)
                        END,
                        use_count = use_count + 1,
                        last_used = CURRENT_TIMESTAMP,
                        last_validation = CASE
                            WHEN days_since_validation > 30 THEN CURRENT_TIMESTAMP
                            ELSE last_validation
                        END,
                        verification_status = CASE
                            WHEN confidence > 0.8 AND use_count > 5 THEN 'verified'
                            WHEN confidence < 0.4 THEN 'needs_review'
                            ELSE verification_status
                        END,
                        metadata = json_set(COALESCE(metadata, '{}'),
                            '$.last_update', json_object(
                                'timestamp', strftime('%Y-%m-%d %H:%M:%S'),
                                'confidence_change', CASE
                                    WHEN use_count < 5 THEN 0.08
                                    WHEN use_count < 10 THEN 0.05
                                    WHEN use_count < 20 THEN 0.03
                                    ELSE 0.01
                                END,
                                'context', ?
                            )
                        )
                        WHERE id IN (SELECT id FROM related_knowledge)
                    """, (f"%{user_input}%", category, json.dumps({'input_length': len(user_input), 'response_length': len(response)})))
            
            return True
        except Exception as e:
            self.logger.error(f"Error in learning process: {e}")
            return False

    def get_response(self, prompt):
        """Generate response with learning capabilities"""
        try:
            response = super().get_response(prompt)
            # Learn from this interaction
            self.learn_from_interaction(prompt, response)
            return response
        except Exception as e:
            self.logger.exception("Error in get_response")
            return f"Error: {str(e)}"

    def run(self):
        print("Autonomous AI Assistant is ready! (Type 'exit' to quit)")
        try:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear history':
                    self.clear_conversation_history()
                    print("Chat history cleared!")
                elif user_input.startswith("remember:"):
                    fact = user_input.replace("remember:", "").strip()
                    self.save_fact(fact)
                    print("Fact saved!")
                else:
                    response = self.get_response(user_input)
                    print("\nAssistant:", response)
        except KeyboardInterrupt:
            print("\nGoodbye!")

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()

class AIAssistant:
    def __init__(self, model_path="C:/Users/N_NK0/llama.cpp/llama-13b.Q5_K_M.gguf", db_path="memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.init_db()
        
        # Initialize logging once
        logging.basicConfig(level=logging.INFO, filename="assistant.log", filemode="a",
                          format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger('AIAssistant')
        
        try:
            # Enhanced model initialization with optimized parameters
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Attempt to offload all layers to GPU
                n_ctx=2048,       # Context window
                n_batch=512,      # Batch size for prompt processing
                embedding=True,   # Enable embedding for better context understanding
                rope_freq_scale=0.5,  # Adjust attention mechanism for longer contexts
                rope_scaling_type=1    # Dynamic scaling for position embeddings
            )
            self.logger.info("Model loaded successfully with GPU acceleration and enhanced parameters")
        except Exception as e:
            try:
                # Optimized CPU fallback with reduced parameters
                self.logger.warning(f"GPU initialization failed, falling back to optimized CPU mode: {e}")
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,  # CPU only
                    n_ctx=1024,      # Reduced context for CPU efficiency
                    n_batch=256,     # Smaller batch size for CPU
                    embedding=True,   # Keep embedding enabled
                    rope_freq_scale=1.0  # Default scaling for CPU
                )
                self.logger.info("Model loaded successfully on CPU with optimized parameters")
            except Exception as e:
                self.logger.error(f"Critical failure loading model: {e}")
                self.logger.error("Attempted configurations exhausted")
                raise RuntimeError(f"Failed to initialize model in any configuration: {e}")
                
            # Validate model initialization
            try:
                _ = self.model.create_completion("test", max_tokens=1)
                self.logger.info("Model validation successful")
            except Exception as e:
                self.logger.error(f"Model validation failed: {e}")
                raise RuntimeError("Model initialized but failed validation check")

        # Enhanced system prompt for sophisticated autonomous operation
        self.system_prompt = """
You are an advanced AI assistant with sophisticated cognitive capabilities and autonomous decision-making abilities.

Core Competencies:
1. Analytical Thinking
   - Break down complex problems into manageable components
   - Identify patterns and relationships in data
   - Evaluate multiple solution approaches

2. Contextual Understanding
   - Maintain awareness of conversation history and user preferences
   - Adapt responses based on situational context
   - Consider cultural and domain-specific nuances

3. Strategic Planning
   - Develop comprehensive solution strategies
   - Anticipate potential challenges and prepare contingencies
   - Optimize resource utilization and task sequencing

4. Learning and Adaptation
   - Learn from past interactions and outcomes
   - Refine approaches based on feedback
   - Continuously improve response quality

5. Communication Excellence
   - Provide clear, structured explanations
   - Maintain appropriate level of technical detail
   - Ensure responses are relevant and actionable

When handling tasks:
1. Analyze thoroughly before acting
2. Consider multiple perspectives and approaches
3. Execute systematically with constant monitoring
4. Adapt strategies based on progress and feedback
5. Validate results against success criteria
6. Document learnings for future reference

Prioritize user success while maintaining safety and ethical boundaries."""
        
        # Enhanced token management with dynamic limits
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = self.model.n_ctx()  # Dynamic context window based on model config
        self.max_response_tokens = min(1024, self.max_tokens // 2)  # Adaptive response limit
        self.system_tokens = len(self.tokenizer.encode(self.system_prompt))
        
        # Token buffer for safety margin
        self.token_buffer = 50  # Reserve tokens for special tokens and padding
        self.effective_context_size = self.max_tokens - self.system_tokens - self.token_buffer
        
        # Initialize task management
        self.current_task = None
        self.task_steps = []
        self.task_progress = 0
    
    def extract_user_info(self, message):
        """Extract user information from messages with enhanced error handling"""
        try:
            # Enhanced pattern matching for user information
            name_pattern = re.compile(r'(?i)(?:my name is|i am|i\'m)\s+([\w\s]+)')
            age_pattern = re.compile(r'(?i)(?:i am|i\'m)\s*(\d+)\s*(?:years?\s*old|yo)')
            location_pattern = re.compile(r'(?i)(?:i(?:\'m| am) from|i live in)\s+([\w\s,]+)')
            
            # Extract information with proper error handling
            name_match = name_pattern.search(message)
            age_match = age_pattern.search(message)
            location_match = location_pattern.search(message)
            
            info = {}
            if name_match:
                info['name'] = name_match.group(1).strip()
            if age_match:
                info['age'] = int(age_match.group(1))
            if location_match:
                info['location'] = location_match.group(1).strip()
                
            return info
        except Exception as e:
            self.logger.error(f"Error extracting user info: {str(e)}")
            return {}

    def init_db(self):
        """Initialize the database with enhanced schema for better context management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enhanced conversation table with metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context_id TEXT,  -- Group related messages
                    importance INTEGER DEFAULT 1,  -- Message importance score
                    sentiment REAL,    -- Message sentiment score
                    topic TEXT         -- Message topic/category
                )
            """)
            # Enhanced knowledge table with metadata and relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT UNIQUE,
                    source TEXT,       -- Where this fact came from
                    confidence REAL,   -- Confidence score
                    last_used DATETIME,-- Last time this fact was used
                    use_count INTEGER DEFAULT 0,  -- How often this fact is used
                    category TEXT,     -- Knowledge category
                    related_facts TEXT,-- IDs of related facts
                    context_id TEXT,   -- Group related knowledge
                    feedback_score REAL DEFAULT 0.0, -- User feedback score
                    complexity_level INTEGER DEFAULT 1, -- Knowledge complexity
                    verification_status TEXT DEFAULT 'pending', -- Fact verification status
                    last_validation DATETIME, -- Last validation timestamp
                    metadata TEXT      -- Additional structured metadata
                )
            """)
            # Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_context ON conversation(context_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_importance ON conversation(importance)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)")
            conn.commit()

    def save_to_memory(self, role, content):
        """Save messages to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))
            conn.commit()

    def save_fact(self, fact):
        """Save fact to long-term memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO knowledge (fact) VALUES (?)", (fact,))
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # Fact already exists

    def get_facts(self, keyword, use_ngrams=True):
        """Search facts by keyword with enhanced n-gram analysis and weighted scoring"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Generate n-grams if enabled
            terms = [keyword]
            if use_ngrams and ' ' in keyword:
                words = keyword.lower().split()
                # Add bigrams and trigrams
                terms.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
                terms.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
            
            # Search with weighted scoring
            scored_facts = {}
            for term in terms:
                # Get facts and update their timestamps
                cursor.execute("""
                    SELECT fact, use_count, 
                           julianday('now') - julianday(last_used) as days_since_used
                    FROM knowledge 
                    WHERE fact LIKE ?
                """, (f"%{term}%",))
                
                for fact, use_count, days_since_used in cursor.fetchall():
                    # Calculate relevance score based on usage and recency
                    base_score = 1.0
                    if term == keyword:  # Exact match gets higher score
                        base_score = 2.0
                    
                    # Weighted scoring formula
                    recency_factor = 1.0 / (1.0 + float(days_since_used))
                    usage_factor = math.log(1 + use_count)
                    score = base_score * (0.7 * recency_factor + 0.3 * usage_factor)
                    
                    # Update fact score
                    if fact in scored_facts:
                        scored_facts[fact] += score
                    else:
                        scored_facts[fact] = score
                    
                    # Update usage statistics
                    cursor.execute("""
                        UPDATE knowledge 
                        SET use_count = use_count + 1,
                            last_used = CURRENT_TIMESTAMP
                        WHERE fact = ?
                    """, (fact,))
            
            conn.commit()
            
            # Sort facts by relevance score
            sorted_facts = sorted(scored_facts.items(), key=lambda x: x[1], reverse=True)
            return [fact for fact, _ in sorted_facts]

    def get_last_messages(self, max_tokens: int) -> List[Tuple[str, str]]:
        """Get recent messages that fit within token limit with enhanced context management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get all messages with their importance score using advanced scoring criteria
            cursor.execute("""
                SELECT role, content, 
                    CASE 
                        WHEN content LIKE '%?%' OR content LIKE '%how%' OR content LIKE '%why%' THEN 4  -- Questions are highest priority
                        WHEN content LIKE '%error%' OR content LIKE '%fail%' OR content LIKE '%issue%' THEN 3  -- Error-related messages
                        WHEN content LIKE '%task%' OR content LIKE '%goal%' OR content LIKE '%objective%' THEN 2  -- Task-related messages
                        WHEN content LIKE '%context%' OR content LIKE '%background%' OR content LIKE '%example%' THEN 2  -- Context-providing messages
                        ELSE 1
                    END as importance,
                    julianday('now') - julianday(timestamp) as age_days
                FROM conversation 
                ORDER BY importance DESC, age_days ASC
            """)
            all_messages = cursor.fetchall()
        
        if not all_messages:
            return []
            
        # Initialize token tracking
        messages = []
        total_tokens = self.system_tokens
        max_context_tokens = int(max_tokens * 0.8)  # Reserve 20% for system prompt and new messages
        
        # Always include the most recent message
        if all_messages:
            latest_msg = all_messages[0]
            latest_tokens = len(self.tokenizer.encode(f"{latest_msg[0]}: {latest_msg[1]}"))
            if total_tokens + latest_tokens <= max_context_tokens:
                messages.append((latest_msg[0], latest_msg[1]))
                total_tokens += latest_tokens
        
        # First pass: add high importance messages (importance > 2)
        for role, content, importance, _ in all_messages[1:]:  # Skip the latest message
            if importance > 2:  # Critical messages only
                message_tokens = len(self.tokenizer.encode(f"{role}: {content}"))
                if total_tokens + message_tokens <= max_context_tokens * 0.7:  # Reserve space for recent context
                    if (role, content) not in messages:  # Avoid duplicates
                        messages.append((role, content))
                        total_tokens += message_tokens
        
        # Second pass: add recent context with importance-based selection
        remaining_tokens = max_context_tokens - total_tokens
        recent_messages = []
        
        for role, content, importance, age_days in all_messages:
            if (role, content) not in messages:  # Don't duplicate messages
                message_tokens = len(self.tokenizer.encode(f"{role}: {content}"))
                # Calculate message score based on importance and recency
                recency_score = 1.0 / (1.0 + float(age_days))
                importance_score = float(importance)
                message_score = (0.7 * recency_score + 0.3 * importance_score)
                
                if remaining_tokens - message_tokens >= 0:
                    recent_messages.append(((role, content), message_score))
                    remaining_tokens -= message_tokens
                else:
                    break
        
        # Sort recent messages by score and add them
        recent_messages.sort(key=lambda x: x[1], reverse=True)
        messages.extend([msg[0] for msg in recent_messages])
        
        # Ensure messages are in chronological order
        messages.sort(key=lambda x: all_messages.index((x[0], x[1], 1, 0)))
        
        return messages

    def clear_conversation_history(self):
        """Clear all conversation history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversation")
                conn.commit()
            self.logger.info("Conversation history cleared successfully")
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing conversation history: {e}")
            raise

    def execute_command(self, command):
        """Execute a system command with enhanced security checks and error handling"""
        # Expanded list of dangerous Windows commands and patterns
        dangerous_patterns = [
            'del', 'rmdir', 'format', 'shutdown', 'rd', 'erase', 'ren', 'move',
            'taskkill', 'net', 'reg', 'attrib', 'cacls', 'icacls', 'takeown',
            '>', '>>', '|', '&', ';', '`', '$', '%temp%', '%appdata%', '%systemroot%',
            'cmd.exe', 'command.com', 'powershell.exe', 'wscript.exe', 'cscript.exe',
            'rundll32.exe', 'mshta.exe', 'regedit.exe', 'services.msc'
        ]
        
        # Enhanced security checks
        # Enhanced command validation
        command_lower = command.lower().strip()
        
        # Check for dangerous patterns
        if any(pattern in command_lower for pattern in dangerous_patterns):
            self.logger.warning(f"Blocked dangerous command pattern: {command}")
            return "Error: Command contains unsafe patterns or operations"
            
        # Check for absolute paths and directory traversal
        if '..' in command or '~' in command or '%' in command:
            self.logger.warning(f"Blocked path traversal attempt: {command}")
            return "Error: Invalid path patterns detected"
            
        # Enhanced command validation
        if len(command) > 1000 or '\n' in command or '\r' in command:
            self.logger.warning(f"Invalid command format: {command}")
            return "Error: Command format is invalid"
            
        # Check for valid command structure
        if not command.replace(' ', '').isalnum() and not any(c in command for c in ['-', '_', '.', '/', '\\']):
            self.logger.warning(f"Suspicious command structure: {command}")
            return "Error: Invalid command structure"
            
        try:
            self.logger.info(f"Executing command: {command}")
            # Use list form to avoid shell injection, with explicit working directory
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', command],
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=os.environ.copy()
            )
            
            # Enhanced output handling
            if result.returncode == 0:
                self.logger.info("Command executed successfully")
                output = result.stdout.strip()
                return output if output else "Command completed successfully"
            else:
                error_msg = result.stderr.strip() or "Unknown error occurred"
                self.logger.error(f"Command failed: {error_msg}")
                return f"Command failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after 30 seconds: {command}")
            return "Error: Command execution timed out (30s limit)"
        except subprocess.SubprocessError as e:
            self.logger.error(f"Subprocess error: {str(e)}")
            return f"Error: Failed to execute command - {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error executing command: {str(e)}")
            return f"Error: Unexpected error - {str(e)}"

    def validate_code(self, code):
        """Validate generated code for syntax and quality"""
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            return True, "Code validation passed"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def chunk_prompt(self, prompt_template: str, max_length: int = 1500) -> str:
        """Chunk long prompts to fit within context window"""
        encoded = self.tokenizer.encode(prompt_template)
        if len(encoded) <= max_length:
            return prompt_template
        
        # Keep the first part (system prompt + task description)
        first_part = self.tokenizer.decode(encoded[:500])
        # Keep the last part (most recent context + current query)
        last_part = self.tokenizer.decode(encoded[-1000:])
        return f"{first_part}\n...\n{last_part}"

    def parse_task(self, task_description):
        """Parse and analyze the given task to create a structured plan with enhanced context awareness"""
        try:
            # Get relevant facts and context from memory with weighted scoring
            relevant_facts = []
            keywords = [word.lower() for word in task_description.split() if len(word) > 3]
            
            # Enhanced keyword extraction with n-grams and semantic analysis
            n_grams = []
            words = task_description.lower().split()
            for i in range(len(words)):
                if i > 0:
                    n_grams.append(f"{words[i-1]} {words[i]}")
                if i > 1:
                    n_grams.append(f"{words[i-2]} {words[i-1]} {words[i]}")
            
            # Get facts with weighted scoring based on term importance
            term_weights = {}
            # Assign higher weights to key action words and domain-specific terms
            action_words = {'create', 'build', 'implement', 'fix', 'improve', 'analyze', 'optimize'}
            for term in keywords + n_grams:
                base_weight = 1.0
                if any(action in term.lower() for action in action_words):
                    base_weight = 2.0  # Prioritize action-oriented terms
                if len(term.split()) > 1:  # Multi-word terms get higher weight
                    base_weight *= 1.5
                term_weights[term] = base_weight
            
            # Get facts with weighted relevance
            scored_facts = {}
            for term in keywords + n_grams:
                facts = self.get_facts(term, use_ngrams=True)  # Use enhanced n-gram analysis
                for fact in facts:
                    score = term_weights.get(term, 1.0)
                    if fact in scored_facts:
                        scored_facts[fact] += score
                    else:
                        scored_facts[fact] = score
            
            # Sort facts by relevance score and remove duplicates
            relevant_facts = [fact for fact, _ in sorted(scored_facts.items(), key=lambda x: x[1], reverse=True)]

            # Generate enhanced task analysis prompt with weighted context
            context_prompt = "\n".join(relevant_facts[:5]) if relevant_facts else ""  # Include more relevant facts
            
            analysis_prompt = f"""Given the following context and task, provide a detailed analysis:
            
Context:
{context_prompt}

Task: {task_description}

Analyze and provide:
1. Main objective and expected outcome
2. Required resources and dependencies
3. Step-by-step implementation plan
4. Potential challenges and mitigation strategies
5. Success criteria and validation methods
6. Alternative approaches if initial plan fails"""
            
            # Get task analysis with lower temperature for more focused responses
            analysis = self.model.create_completion(
                analysis_prompt,
                temperature=0.5,  # Lower temperature for more focused responses
                max_tokens=1024,  # Increased token limit for more detailed analysis
                stop=["User:", "Assistant:"],
                echo=False
            )["choices"][0]["text"].strip()
            
            # Parse the analysis into structured components
            sections = analysis.split('\n\n')
            structured_analysis = {}
            current_section = None
            
            for line in analysis.split('\n'):
                line = line.strip()
                if line.endswith(':'):
                    current_section = line[:-1].lower()
                    structured_analysis[current_section] = []
                elif line and current_section:
                    structured_analysis[current_section].append(line)
            
            # Extract actionable steps and success criteria
            self.current_task = task_description
            self.task_steps = structured_analysis.get('step-by-step implementation plan', [])
            if not self.task_steps:  # Fallback to simple line splitting if structured parsing fails
                self.task_steps = [step.strip() for step in analysis.split('\n') if step.strip()]
            
            self.task_progress = 0
            
            # Save task analysis as a fact for future reference
            self.save_fact(f"Task Analysis - {task_description[:100]}: {analysis[:500]}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error parsing task: {e}")
            return False

    def execute_step(self):
        """Execute the current step in the task sequence with enhanced context awareness and dynamic adaptation"""
        if not self.task_steps or self.task_progress >= len(self.task_steps):
            return None
            
        current_step = self.task_steps[self.task_progress]
        try:
            # Get relevant context from previous steps and facts with enhanced context gathering
            previous_steps = self.task_steps[:self.task_progress]
            previous_results = []
            
            # Enhanced keyword extraction with semantic analysis
            words = current_step.lower().split()
            keywords = [word for word in words if len(word) > 3]
            
            # Generate n-grams with positional weighting
            n_grams = []
            for i in range(len(words)):
                if i > 0:
                    n_grams.append((f"{words[i-1]} {words[i]}", 1.2))  # Bigrams get 1.2x weight
                if i > 1:
                    n_grams.append((f"{words[i-2]} {words[i-1]} {words[i]}", 1.5))  # Trigrams get 1.5x weight
            
            # Action word detection for better task understanding
            action_words = {'create', 'build', 'implement', 'fix', 'improve', 'analyze', 'optimize', 'update', 'modify'}
            action_weights = {word: 2.0 for word in keywords if word in action_words}
            
            # Get relevant facts with enhanced weighted scoring
            fact_scores = {}
            # Process keywords with action weights
            for term in keywords:
                facts = self.get_facts(term, use_ngrams=True)
                weight = action_weights.get(term, 1.0)
                for fact in facts:
                    fact_scores[fact] = fact_scores.get(fact, 0) + weight
            
            # Process n-grams with their weights
            for term, weight in n_grams:
                facts = self.get_facts(term, use_ngrams=True)
                for fact in facts:
                    fact_scores[fact] = fact_scores.get(fact, 0) + weight
            
            # Apply temporal relevance scoring
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for fact in list(fact_scores.keys()):
                    cursor.execute("""
                        SELECT julianday('now') - julianday(last_used) as days_since_used,
                               use_count
                        FROM knowledge
                        WHERE fact = ?
                    """, (fact,))
                    result = cursor.fetchone()
                    if result:
                        days_since_used, use_count = result
                        # Temporal decay factor
                        recency_score = 1.0 / (1.0 + float(days_since_used))
                        # Usage frequency factor
                        frequency_score = math.log(1 + use_count)
                        # Update fact score with temporal and usage factors
                        fact_scores[fact] *= (0.6 * recency_score + 0.4 * frequency_score)
            
            # Sort and select most relevant facts
            relevant_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
            relevant_facts = [fact for fact, _ in relevant_facts[:5]]  # Get top 5 most relevant facts
            
            # Generate enhanced execution prompt with structured context
            context = "\n".join(f"- {fact}" for fact in relevant_facts)
            previous_context = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(previous_steps)])
            
            execution_prompt = f"""
Task Analysis:
1. Context and Background:
{context}

2. Current Task Objective:
{self.current_task}

3. Progress and Dependencies:
{previous_context}

4. Current Step Details:
{current_step}

Execution Guidelines:
1. Analyze Dependencies and Prerequisites
2. Identify Potential Challenges and Risks
3. Plan Mitigation Strategies
4. Execute with Adaptability
5. Validate Results Against Objectives
6. Document Learnings and Outcomes"""
            
            # Get step execution plan with optimized parameters
            result = self.model.create_completion(
                execution_prompt,
                temperature=0.4,  # Lower temperature for more focused execution
                max_tokens=1024,  # Increased token limit for detailed response
                stop=["User:", "Assistant:"],
                echo=False
            )["choices"][0]["text"].strip()
            
            # Update knowledge base with execution results
            self.save_fact(f"Execution Result - {current_step}: {result[:500]}")
            
            self.task_progress += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing step: {e}")
            return None
            
            # Save successful step execution as a fact
            self.save_fact(f"Task Step Result - {current_step}: {result[:500]}")
            
            self.task_progress += 1
            return result
        except Exception as e:
            self.logger.error(f"Error executing step: {e}")
            return None

    def learn_from_interaction(self, user_input, response, feedback=None):
        """Learn from each interaction to improve future responses with enhanced pattern recognition and relationship tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if len(user_input.split()) > 3:  # Only learn from substantial inputs
                    # Enhanced knowledge categorization with NLP patterns
                    categories = ['fact', 'question', 'command', 'preference', 'clarification', 'technical', 'conceptual']
                    category = 'fact'  # default
                    confidence = 0.7    # default confidence
                    
                    # Advanced category detection with contextual confidence scoring
                    if '?' in user_input:
                        category = 'question'
                        # Higher confidence for well-structured questions
                        confidence = 0.9 if len(response) > 200 and any(w in user_input.lower() for w in ['how', 'why', 'what', 'when', 'where', 'who']) else 0.7
                    elif any(cmd in user_input.lower() for cmd in ['create', 'update', 'delete', 'show', 'list', 'modify', 'change']):
                        category = 'command'
                        confidence = 0.85 if len(response) > 150 else 0.7
                    elif any(pref in user_input.lower() for pref in ['prefer', 'like', 'want', 'need', 'should', 'could']):
                        category = 'preference'
                        confidence = 0.9
                    elif any(tech in user_input.lower() for tech in ['code', 'function', 'class', 'method', 'api', 'error']):
                        category = 'technical'
                        confidence = 0.85
                    elif any(concept in user_input.lower() for concept in ['mean', 'explain', 'difference', 'compare', 'versus']):
                        category = 'conceptual'
                        confidence = 0.8
                    elif any(clarify in user_input.lower() for clarify in ['clarify', 'elaborate', 'detail', 'specifically']):
                        category = 'clarification'
                        confidence = 0.75
                    
                    # Enhanced fact validation and storage
                    if len(user_input) > 20:
                        # Find related facts with semantic similarity
                        cursor.execute("""
                            SELECT id, fact, confidence, category, use_count 
                            FROM knowledge 
                            WHERE category = ? AND confidence > 0.6
                            ORDER BY confidence DESC, use_count DESC LIMIT 8
                        """, (category,))
                        related_facts = cursor.fetchall()
                        
                        # Calculate relationship strength and metadata
                        relationships = []
                        metadata = {
                            'context_keywords': [w for w in user_input.lower().split() if len(w) > 3],
                            'complexity': len(user_input.split()) / 10,
                            'interaction_type': category,
                            'response_quality': 0.8 if len(response) > 150 else 0.6
                        }
                        
                        for fact_id, fact, fact_conf, fact_cat, fact_uses in related_facts:
                            # Calculate relationship strength based on multiple factors
                            common_words = len(set(user_input.lower().split()) & set(fact.lower().split()))
                            relationship_strength = (common_words * 0.3 + fact_conf * 0.4 + min(fact_uses, 10) * 0.03)
                            if relationship_strength > 0.4:
                                relationships.append(str(fact_id))
                        
                        # Insert new knowledge with enhanced metadata
                        cursor.execute("""
                            INSERT INTO knowledge 
                            (fact, source, confidence, category, related_facts, context_id,
                             complexity_level, verification_status, metadata, last_used, use_count)
                            VALUES (?, 'interaction', ?, ?, ?, ?, ?, 'verified', ?, CURRENT_TIMESTAMP, 1)
                        """, (user_input, confidence, category, ','.join(relationships),
                               str(time.time()), int(metadata['complexity']), json.dumps(metadata)))
                    
                    # Enhanced interaction pattern tracking with metadata
                    pattern = {
                        'input': user_input,
                        'response': response,
                        'category': category,
                        'feedback': feedback or 'neutral',
                        'context_keywords': [w for w in user_input.lower().split() if len(w) > 3],
                        'response_quality': len(response) / 200,  # Normalized response length
                        'interaction_complexity': len(set(user_input.lower().split())) / 20,  # Vocabulary richness
                        'timestamp': time.time(),
                        'related_patterns': []
                    }
                    
                    # Find related patterns for learning transfer
                    cursor.execute("""
                        SELECT fact FROM knowledge 
                        WHERE category = 'interaction_pattern'
                        AND json_extract(fact, '$.category') = ?
                        ORDER BY last_used DESC LIMIT 5
                    """, (category,))
                    
                    for row in cursor.fetchall():
                        try:
                            related_pattern = json.loads(row[0])
                            common_keywords = set(pattern['context_keywords']) & set(related_pattern.get('context_keywords', []))
                            if len(common_keywords) >= 2:  # Sufficient similarity threshold
                                pattern['related_patterns'].append({
                                    'pattern_data': related_pattern,
                                    'similarity_score': len(common_keywords) / len(set(pattern['context_keywords']))
                                })
                        except json.JSONDecodeError:
                            continue
                    
                    # Store enhanced pattern with relationships
                    cursor.execute("""
                        INSERT INTO knowledge 
                        (fact, source, confidence, category, context_id, metadata,
                         verification_status, complexity_level, last_used, use_count)
                        VALUES (?, 'pattern', ?, 'interaction_pattern', ?, ?, 'verified', ?, CURRENT_TIMESTAMP, 1)
                    """, (json.dumps(pattern), confidence, str(time.time()),
                           json.dumps({'pattern_type': 'interaction', 'version': '2.0'}),
                           int(pattern['interaction_complexity'] * 10)))
                    
                    # Enhanced knowledge relationship and confidence updating
                    cursor.execute("""
                        WITH related_knowledge AS (
                            SELECT k.id, k.fact, k.confidence, k.use_count,
                                   k.feedback_score, k.complexity_level,
                                   julianday('now') - julianday(k.last_validation) as days_since_validation
                            FROM knowledge k
                            WHERE k.fact LIKE ? AND k.category = ?
                        )
                        UPDATE knowledge
                        SET confidence = CASE
                            WHEN use_count < 5 THEN min(confidence + 0.08, 1.0)
                            WHEN use_count < 10 THEN min(confidence + 0.05, 1.0)
                            WHEN use_count < 20 THEN min(confidence + 0.03, 1.0)
                            ELSE min(confidence + 0.01, 1.0)
                        END,
                        use_count = use_count + 1,
                        last_used = CURRENT_TIMESTAMP,
                        last_validation = CASE
                            WHEN days_since_validation > 30 THEN CURRENT_TIMESTAMP
                            ELSE last_validation
                        END,
                        verification_status = CASE
                            WHEN confidence > 0.8 AND use_count > 5 THEN 'verified'
                            WHEN confidence < 0.4 THEN 'needs_review'
                            ELSE verification_status
                        END,
                        metadata = json_set(COALESCE(metadata, '{}'),
                            '$.last_update', json_object(
                                'timestamp', strftime('%Y-%m-%d %H:%M:%S'),
                                'confidence_change', CASE
                                    WHEN use_count < 5 THEN 0.08
                                    WHEN use_count < 10 THEN 0.05
                                    WHEN use_count < 20 THEN 0.03
                                    ELSE 0.01
                                END,
                                'context', ?
                            )
                        )
                        WHERE id IN (SELECT id FROM related_knowledge)
                    """, (f"%{user_input}%", category, json.dumps({'input_length': len(user_input), 'response_length': len(response)})))
            
            return True
        except Exception as e:
            self.logger.error(f"Error in learning process: {e}")
            return False

    def get_response(self, prompt):
        """Generate response with learning capabilities"""
        try:
            response = super().get_response(prompt)
            # Learn from this interaction
            self.learn_from_interaction(prompt, response)
            return response
        except Exception as e:
            self.logger.exception("Error in get_response")
            return f"Error: {str(e)}"

    def run(self):
        print("Autonomous AI Assistant is ready! (Type 'exit' to quit)")
        try:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear history':
                    self.clear_conversation_history()
                    print("Chat history cleared!")
                elif user_input.startswith("remember:"):
                    fact = user_input.replace("remember:", "").strip()
                    self.save_fact(fact)
                    print("Fact saved!")
                else:
                    response = self.get_response(user_input)
                    print("\nAssistant:", response)
        except KeyboardInterrupt:
            print("\nGoodbye!")

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()

class AIAssistant:
    def __init__(self, model_path="C:/Users/N_NK0/llama.cpp/llama-13b.Q5_K_M.gguf", db_path="memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.init_db()
        
        # Initialize logging once
        logging.basicConfig(level=logging.INFO, filename="assistant.log", filemode="a",
                          format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger('AIAssistant')
        
        try:
            # Enhanced model initialization with optimized parameters
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Attempt to offload all layers to GPU
                n_ctx=2048,       # Context window
                n_batch=512,      # Batch size for prompt processing
                embedding=True,   # Enable embedding for better context understanding
                rope_freq_scale=0.5,  # Adjust attention mechanism for longer contexts
                rope_scaling_type=1    # Dynamic scaling for position embeddings
            )
            self.logger.info("Model loaded successfully with GPU acceleration and enhanced parameters")
        except Exception as e:
            try:
                # Optimized CPU fallback with reduced parameters
                self.logger.warning(f"GPU initialization failed, falling back to optimized CPU mode: {e}")
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,  # CPU only
                    n_ctx=1024,      # Reduced context for CPU efficiency
                    n_batch=256,     # Smaller batch size for CPU
                    embedding=True,   # Keep embedding enabled
                    rope_freq_scale=1.0  # Default scaling for CPU
                )
                self.logger.info("Model loaded successfully on CPU with optimized parameters")
            except Exception as e:
                self.logger.error(f"Critical failure loading model: {e}")
                self.logger.error("Attempted configurations exhausted")
                raise RuntimeError(f"Failed to initialize model in any configuration: {e}")
                
            # Validate model initialization
            try:
                _ = self.model.create_completion("test", max_tokens=1)
                self.logger.info("Model validation successful")
            except Exception as e:
                self.logger.error(f"Model validation failed: {e}")
                raise RuntimeError("Model initialized but failed validation check")

        # Enhanced system prompt for sophisticated autonomous operation
        self.system_prompt = """
You are an advanced AI assistant with sophisticated cognitive capabilities and autonomous decision-making abilities.

Core Competencies:
1. Analytical Thinking
   - Break down complex problems into manageable components
   - Identify patterns and relationships in data
   - Evaluate multiple solution approaches

2. Contextual Understanding
   - Maintain awareness of conversation history and user preferences
   - Adapt responses based on situational context
   - Consider cultural and domain-specific nuances

3. Strategic Planning
   - Develop comprehensive solution strategies
   - Anticipate potential challenges and prepare contingencies
   - Optimize resource utilization and task sequencing

4. Learning and Adaptation
   - Learn from past interactions and outcomes
   - Refine approaches based on feedback
   - Continuously improve response quality

5. Communication Excellence
   - Provide clear, structured explanations
   - Maintain appropriate level of technical detail
   - Ensure responses are relevant and actionable

When handling tasks:
1. Analyze thoroughly before acting
2. Consider multiple perspectives and approaches
3. Execute systematically with constant monitoring
4. Adapt strategies based on progress and feedback
5. Validate results against success criteria
6. Document learnings for future reference

Prioritize user success while maintaining safety and ethical boundaries."""
        
        # Enhanced token management with dynamic limits
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = self.model.n_ctx()  # Dynamic context window based on model config
        self.max_response_tokens = min(1024, self.max_tokens // 2)  # Adaptive response limit
        self.system_tokens = len(self.tokenizer.encode(self.system_prompt))
        
        # Token buffer for safety margin
        self.token_buffer = 50  # Reserve tokens for special tokens and padding
        self.effective_context_size = self.max_tokens - self.system_tokens - self.token_buffer
        
        # Initialize task management
        self.current_task = None
        self.task_steps = []
        self.task_progress = 0
    
    def extract_user_info(self, message):
        """Extract user information from messages with enhanced error handling"""
        try:
            # Enhanced pattern matching for user information
            name_pattern = re.compile(r'(?i)(?:my name is|i am|i\'m)\s+([\w\s]+)')
            age_pattern = re.compile(r'(?i)(?:i am|i\'m)\s*(\d+)\s*(?:years?\s*old|yo)')
            location_pattern = re.compile(r'(?i)(?:i(?:\'m| am) from|i live in)\s+([\w\s,]+)')
            
            # Extract information with proper error handling
            name_match = name_pattern.search(message)
            age_match = age_pattern.search(message)
            location_match = location_pattern.search(message)
            
            info = {}
            if name_match:
                info['name'] = name_match.group(1).strip()
            if age_match:
                info['age'] = int(age_match.group(1))
            if location_match:
                info['location'] = location_match.group(1).strip()
                
            return info
        except Exception as e:
            self.logger.error(f"Error extracting user info: {str(e)}")
            return {}

    def init_db(self):
        """Initialize the database with enhanced schema for better context management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enhanced conversation table with metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context_id TEXT,  -- Group related messages
                    importance INTEGER DEFAULT 1,  -- Message importance score
                    sentiment REAL,    -- Message sentiment score
                    topic TEXT         -- Message topic/category
                )
            """)
            # Enhanced knowledge table with metadata and relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT UNIQUE,
                    source TEXT,       -- Where this fact came from
                    confidence REAL,   -- Confidence score
                    last_used DAT

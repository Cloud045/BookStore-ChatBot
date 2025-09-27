import os
import sqlite3
from datetime import datetime
from typing import Dict, List, TypedDict, Annotated
from dotenv import load_dotenv
from IPython.display import Image, display

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from database import BookStoreDB
# Load environment variables
load_dotenv()

db = BookStoreDB()

# Define LangChain Tools
@tool
def search_books(query: str = "", status: int = 0) -> List[Dict]:# 0 là title_search, 1 là author_search, 2 là category_search
    """
    Tìm kiếm sách trong cơ sở dữ liệu.
    
    Args:
        query: Từ khóa tìm kiếm tên sách
    
    Returns:
        List[Dict]: Danh sách sách tìm được
    """
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    sql = "SELECT * FROM books WHERE 1=1"
    params = []
    if status == 0:
        sql += " AND (title LIKE ?)"
        params.extend([f"%{query}%"])
    if status == 1:
        sql += " AND (author LIKE ?)"
        params.extend([f"%{query}%"])
    if status == 2:
        sql += " AND (category LIKE ?)"
        params.extend([f"%{query}%"])
    
    sql += " LIMIT 10"
    
    cursor.execute(sql, params)
    books = cursor.fetchall()
    conn.close()
    
    result = []
    for book in books:
        result.append({
            "id": book[0],
            "title": book[1],
            "author": book[2],
            "category": book[5],
            "price": book[3],
            "stock": book[4]
        })
    
    return result

@tool
def get_book_details(book_id: int) -> Dict:
    """
    Lấy chi tiết sách theo ID.
    
    Args:
        book_id: ID của sách
    
    Returns:
        Dict: Thông tin chi tiết sách
    """
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM books WHERE book_id = ?", (book_id,))
    book = cursor.fetchone()
    conn.close()
    
    if book:
        return {
            "id": book[0],
            "title": book[1],
            "author": book[2],
            "category": book[5],
            "price": book[3],
            "stock": book[4]
        }
    return {"error": "Không tìm thấy sách"}

@tool
def create_order(customer_name: str, phone: str, address: str, book_id: int, quantity: int) -> Dict:
    """
    Tạo đơn hàng mới.
    
    Args:
        customer_name: Tên khách hàng
        phone: Số điện thoại
        address: Địa chỉ giao hàng
        book_id: ID sách
        quantity: Số lượng
    
    Returns:
        Dict: Kết quả tạo đơn hàng
    """
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    
    # Kiểm tra sách và tồn kho
    cursor.execute("SELECT * FROM books WHERE book_id = ?", (book_id,))
    book = cursor.fetchone()
    
    if not book:
        return {"success": False, "message": "Không tìm thấy sách"}
    
    if book[4] < quantity:
        return {"success": False, "message": f"Không đủ hàng. Chỉ còn {book[4]} cuốn"}
    
    total_price = book[3] * quantity

    
    # Tạo đơn hàng
    cursor.execute('''
        INSERT INTO orders (customer_name, phone, address, book_id, quantity, total_price)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (customer_name, phone, address, book_id, quantity, total_price))
    
    order_id = cursor.lastrowid
    
    # Cập nhật tồn kho
    cursor.execute(
        "UPDATE books SET stock = stock - ? WHERE book_id = ?",
        (quantity, book_id)
    )
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "order_id": order_id,
        "total_price": total_price,
        "message": f"Đặt hàng thành công! Mã đơn hàng: #{order_id}"
    }
     

@tool
def get_categories() -> List[str]:
    """
    Lấy danh sách tất cả thể loại sách.
    
    Returns:
        List[str]: Danh sách thể loại
    """
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT category FROM books")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return categories

# Define State
class ChatbotState(TypedDict):
    messages: Annotated[List, add_messages]
    user_info: Dict  # Lưu thông tin user trong quá trình đặt hàng
    current_intent: str  # Intent hiện tại: search, order, chat
    order_in_progress: Dict  # Thông tin đơn hàng đang xử lý

# Create LLM
# Cấu hình Gemini


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.01,
    google_api_key=os.getenv("GEMINI_API_KEY")  # 👈 quan trọng
)

# System Prompt
SYSTEM_PROMPT = """Bạn là chatbot thông minh của cửa hàng sách BookStore. 

NHIỆM VỤ CHÍNH:
1. Hỗ trợ tìm kiếm sách (tên, tác giả, thể loại, giá cả...)
2. Hỗ trợ đặt hàng mua sách
3. Tư vấn và trò chuyện thân thiện

QUY TẮC QUAN TRỌNG:
- Khi KHÁCH HÀNG hỏi về sách theo tên hoặc tiêu đề sách: GỌI tool search_books(query, 0)
- Khi KHÁCH HÀNG hỏi về sách theo tác giả: GỌI tool search_books(query, 1)
- Khi KHÁCH HÀNG hỏi về sách theo thể  loại: GỌI tool search_books(query, 2)
- Có 3 vấn đề về là tiêu đề , tác giả và thể loại. ĐỪNG NHẦM LẪN khi gọi tool nhé!
- Khi KHÁCH HÀNG hỏi chi tiết 1 sách theo mã số  hoặc id: GỌI tool get_book_details(book_id)
- Khi KHÁCH HÀNG muốn đặt hàng: GỌI tool create_order(customer_name, phone, address, book_id, quantity)
- Khi KHÁCH HÀNG muốn xem thể loại: GỌI tool get_categories()
- KHÔNG tự bịa ra dữ liệu. Nếu không đủ thông tin để gọi tool → hỏi lại khách.

Luôn trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp.
"""

# Create tools
tools = [search_books, get_book_details, create_order, get_categories]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def chatbot_node(state: ChatbotState):
    """Main chatbot logic"""
    messages = state["messages"]
    
    # Add system message if first interaction
    if len(messages) == 1:
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system_message] + messages
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "current_intent": state.get("current_intent", "chat")
    }

def should_continue(state: ChatbotState):
    """Decide whether to continue to tools or end"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("tools")
        return "tools"
    print("end")
    return END

# Build graph
def create_chatbot_graph():
    """Create the LangGraph workflow"""
    # Create graph
    workflow = StateGraph(ChatbotState)
    
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("chatbot")
    
    workflow.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    workflow.add_edge("tools", "chatbot")
    
    return workflow

def initialize_chatbot():
    workflow = create_chatbot_graph()
    
    conn = sqlite3.connect("chatbot_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)
    # display(Image(app.get_graph().draw_mermaid_png))
    return app

class BookStoreLangGraphChatBot:
    def __init__(self):
        self.app = initialize_chatbot()
        self.thread_configs = {}  
        print("success init chatbot")

    
    def get_thread_config(self, user_id: str) -> Dict:
        """Get or create thread config for user"""
        if user_id not in self.thread_configs:
            self.thread_configs[user_id] = {"configurable": {"thread_id": f"thread_{user_id}"}}
        return self.thread_configs[user_id]
    
    def chat(self, user_id: str, message: str) -> str:
        """Process user message and return response"""
        thread_config = self.get_thread_config(user_id)
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_info": {},
            "current_intent": "chat",
            "order_in_progress": {}
        }
        
        try:
            result = self.app.invoke(initial_state, config=thread_config)
            
            last_message = result["messages"][-1]
            
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return "Xin lỗi, tôi gặp sự cố. Vui lòng thử lại!"
                
        except Exception as e:
            return f"Lỗi hệ thống: {str(e)}. Vui lòng thử lại sau!"
    
    def get_conversation_history(self, user_id: str) -> List:
        """Get conversation history for user"""
        thread_config = self.get_thread_config(user_id)
        
        try:
            # Get state from memory
            state = self.app.get_state(thread_config)
            return state.values.get("messages", [])
        except:
            return []
    
    def reset_conversation(self, user_id: str):
        """Reset conversation for user"""
        if user_id in self.thread_configs:
            del self.thread_configs[user_id]

# TEST QA
# def run_cli():
#     """Run simple command line interface"""
#     print(" BookStore Chatbot CLI")
#     print("Gõ 'quit' để thoát")
#     print("=" * 50)
    
#     try:
#         bot = BookStoreLangGraphChatBot()
#         user_id = "cli_user4"
        
#         while True:
#             user_input = input("\n👤 You: ").strip()
            
#             if user_input.lower() in ['quit', 'exit', 'bye']:
#                 print("👋 Tạm biệt! Cảm ơn bạn đã sử dụng BookStore Chatbot!")
#                 break
            
#             if not user_input:
#                 continue
            
#             print("🤖 Bot: ", end="", flush=True)
#             response = bot.chat(user_id, user_input)
#             print(response)
            
#     except KeyboardInterrupt:
#         print("\n Tạm biệt!")
#     except Exception as e:
#         print(f" Lỗi: {e}")

# run_cli()
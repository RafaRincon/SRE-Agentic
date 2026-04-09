from app.indexer.chunker import chunk_file


C_SHARP_CODE = """
using System;
using System.Threading.Tasks;

namespace Ordering.API.Controllers
{
    public class OrdersController
    {
        private readonly IOrderRepository _repo;

        public OrdersController(IOrderRepository repo)
        {
            _repo = repo;
        }

        public async Task<IActionResult> CreateOrder(OrderDraft draft)
        {
            if (draft == null) throw new ArgumentNullException();

            var order = new Order();
            await _repo.SaveAsync(order);

            return Ok();
        }
    }
}
"""


def test_chunker_generates_chunks_with_expected_metadata():
    chunks = chunk_file(
        "src/Ordering.API/Controllers/OrdersController.cs",
        C_SHARP_CODE,
    )

    assert chunks
    assert all(chunk["file_path"].endswith("OrdersController.cs") for chunk in chunks)
    assert all(chunk["service_name"] == "Ordering.API" for chunk in chunks)
    assert all("chunk_text" in chunk and chunk["chunk_text"] for chunk in chunks)
    assert any("CreateOrder" in chunk["chunk_text"] for chunk in chunks)
